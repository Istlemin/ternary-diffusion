import copy
import math
import torch
from torch import nn
from typing import Optional, Tuple

from model.unet import UNet

NO_QUANT = {
    "weight_quanter": None,
    "act_quanter": None,
}

def clip_and_save(ctx, w, clip_val):
    ctx.save_for_backward((w<-clip_val) | (w>clip_val))
    return torch.clamp(w,-clip_val,clip_val)

def gradient_apply_clipping(ctx, grad_output):
    clip_mask, = ctx.saved_tensors
    return grad_output * (~clip_mask)

class TwnQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w : torch.Tensor, dim, clip_val):
        w = clip_and_save(ctx, w, clip_val)

        if dim is None:
            dim = tuple(range(len(w.shape)))
        if type(dim) is int:
            dim = (dim,)
        n = math.prod((w.shape[d]) for d in dim)


        thres = torch.norm(w, p=1, dim=dim) / n * 0.7
        for d in dim:
            thres = thres.unsqueeze(d)

        b = (w>thres).type(w.dtype) - (w<-thres).type(w.dtype)
        alpha = torch.sum(torch.abs(b*w),dim=dim)/torch.sum(torch.abs(b),dim=dim)
        for d in dim:
            alpha = alpha.unsqueeze(d)

        return alpha*b

    @staticmethod
    def backward(ctx, grad_output):
        """
        Approximate the gradient wrt to the full-precision inputs
        using the gradient wrt to the quantized inputs, 
        zeroing out gradient for clipped values.
        """
        # Need to return one gradient for each argument,
        # but we only want one for [w] 
        return gradient_apply_clipping(ctx, grad_output), None, None

class TwnQuantizer(nn.Module):
    def __init__(self, clip_val=2.5):
        super().__init__()
        self.clip_val = clip_val

    def forward(self,w, dim=None):
        return TwnQuantizerFunction.apply(w,dim,self.clip_val)


class MinMaxQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w : torch.Tensor, dim, bits, clip_val):
        w = clip_and_save(ctx, w, clip_val)

        if dim is None:
            dim = tuple(range(len(w.shape)))
        if type(dim) is int:
            dim = (dim,)
        
        mn = mx = w
        for d in dim:
            mn = torch.min(mn,dim=d).values
            mx = torch.max(mx,dim=d).values
            mn = mn.unsqueeze(d)
            mx = mx.unsqueeze(d)

        alpha = (mx-mn + 1e-8)
        size = (2**bits-1)
        quant_w = torch.round((w-mn)/alpha*size)/size*alpha+mn

        return quant_w


    @staticmethod
    def backward(ctx, grad_output):
        """
        Approximate the gradient wrt to the full-precision inputs
        using the gradient wrt to the quantized inputs, 
        zeroing out gradient for clipped values.
        """
        # Need to return one gradient for each argument,
        # but we only want one for [w] 
        #print("back!!")
        return gradient_apply_clipping(ctx, grad_output), None, None, None

class MinMaxQuantizer(nn.Module):
    def __init__(self, bits=8, clip_val=2.5):
        super().__init__()
        self.clip_val = clip_val
        self.bits=bits

    def forward(self,w, dim=None):
        return MinMaxQuantizerFunction.apply(w,dim,self.bits,self.clip_val)


class NoopQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,w,dim=None):
        return w


class QuantizedLinear(nn.Module):
    def __init__(self, linear : nn.Linear, quant_config=NO_QUANT):
        super().__init__()
        
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_quanter = quant_config["weight_quanter"]
        self.act_quanter = quant_config["act_quanter"]

    def forward(self, input, input_quantize_dim=None):
        if self.act_quanter is not None:
            input = self.act_quanter(input,input_quantize_dim)
        quant_weight = self.weight_quanter(self.weight)
        return nn.functional.linear(input, quant_weight, self.bias)
    

class QuantizedConv2d(nn.Module):
    def __init__(self, conv2d : nn.Conv2d, quant_config=NO_QUANT):
        super().__init__()
        
        self.weight = conv2d.weight
        self.bias = conv2d.bias
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.dilation = conv2d.dilation
        self.weight_quanter = quant_config["weight_quanter"]
        self.act_quanter = quant_config["act_quanter"]

    def forward(self, input, input_quantize_dim=None):
        if self.act_quanter is not None:
            input = self.act_quanter(input,input_quantize_dim)
        quant_weight = self.weight_quanter(self.weight, (1,2,3))
        return nn.functional.conv2d(input, quant_weight, self.bias,stride=self.stride, padding=self.padding, dilation=self.dilation)
    
def quantize_residual_block(block,quant_config):
    block.conv1 = QuantizedConv2d(block.conv1,quant_config)
    block.conv2 = QuantizedConv2d(block.conv2,quant_config)
    if isinstance(block.residual_conv, nn.Conv2d):
        block.residual_conv = QuantizedConv2d(block.residual_conv,quant_config)
    

def quantize_attention(attention,quant_config):
    attention.to_qkv = QuantizedConv2d(attention.to_qkv,quant_config)
    attention.to_out = QuantizedConv2d(attention.to_out,quant_config)

def quantize_unet(model, quant_config=None):
    model = copy.deepcopy(model)
    if quant_config is None:
        quant_config = {
            "weight_quanter": TwnQuantizer(clip_val=2.5),
            "act_quanter": NoopQuantizer(),
        }
    
    model.time_embedding[0] = model.time_embedding[0]
    model.time_embedding[2] = model.time_embedding[2]
    
    # model.init_conv = QuantizedConv2d(model.init_conv,quant_config)
    
    for down_block in model.down_blocks:
        quantize_residual_block(down_block[0],quant_config)
        quantize_residual_block(down_block[1],quant_config)
        if isinstance(down_block[2],nn.Identity):
            down_block[3][1] = QuantizedConv2d(down_block[3][1],quant_config)
        else:
            quantize_attention(down_block[2].fn.fn,quant_config)
            down_block[3] = QuantizedConv2d(down_block[3],quant_config)
        
    quantize_residual_block(model.mid_block1,quant_config)
    quantize_attention(model.mid_attn.fn.fn,quant_config)
    quantize_residual_block(model.mid_block2,quant_config)
    
    for up_block in model.up_blocks:
        quantize_residual_block(up_block[0],quant_config)
        quantize_residual_block(up_block[1],quant_config)
        if isinstance(up_block[2],nn.Identity):
            up_block[3][1] = QuantizedConv2d(up_block[3][1],quant_config)
        else:
            quantize_attention(up_block[2].fn.fn,quant_config)
            up_block[3] = QuantizedConv2d(up_block[3],quant_config)
            
    quantize_residual_block(model.out_block,quant_config)
    
    # model.conv_out = QuantizedConv2d(model.conv_out,quant_config)
    
    quantized_param_names = [name+".weight" for name,module in list(model.named_modules()) if isinstance(module,QuantizedConv2d) or isinstance(module, QuantizedLinear)]
    return model, quantized_param_names

def calc_model_size(model, quantized_param_names):
    quant_size = 0
    other_size = 0 
    others = []
    for name, parameter in model.named_parameters():
        if name in quantized_param_names:
            quant_size += 2*parameter.numel()
        else:
            other_size += 16*parameter.numel()
            others.append((parameter.numel(),name))
    others.sort()
    print(others[-5:])
    print(f"Quant size: {quant_size/1000:.3f} kB")
    print(f"Other size: {other_size/1000:.3f} kB")
    print(f"Total size: {(quant_size+other_size)/1000:.3f} kB")
    
if __name__=="__main__":
    model = UNet(3,
                 image_size=64,
                 hidden_dims=[16, 32, 64, 128],
                 use_linear_attn=False)
    
    model, quantized_param_names = quantize_unet(model)
    
    calc_model_size(model,quantized_param_names)