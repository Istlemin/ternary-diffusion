import argparse
from collections import defaultdict
import numpy

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchaudio

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
import torchvision
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

from scheduler import DDIMScheduler
from model import UNet
from utils import save_images, normalize_to_neg_one_to_one, plot_losses
from dataset import CustomDataset
from quantization import calc_model_size, quantize_unet, QuantizedConv2d, QuantizedLinear
import pandas as pd
import math

n_timesteps = 1000
n_inference_timesteps = 50


def main(args):
    model = UNet(3,
                 image_size=args.resolution,
                 hidden_dims=[16, 32, 64, 128],
                 use_linear_attn=False)
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)["model_state"]
        model.load_state_dict(pretrained, strict=False)

    if args.quantize:
        teacher = model
        model, quantized_param_names = quantize_unet(teacher,args)
        
        print("Normal model size:")
        calc_model_size(teacher,[])
        print("Quant model size:")
        calc_model_size(model,quantized_param_names)
        
        #model.load_state_dict(torch.load("trained_models/ddpm-flowers-twn.pth")["model_state"])
    
    distillation_transforms = torch.nn.ModuleDict({
        "3": torch.nn.Linear(3,3),
        "4": torch.nn.Linear(4,4),
        "8": torch.nn.Linear(8,8),
        "16": torch.nn.Linear(16,16),
        "32": torch.nn.Linear(32,32),
        "64": torch.nn.Linear(64,64),
        "128": torch.nn.Linear(128,128),
    })
    
    optimizer = torch.optim.Adam(
        [
            {'params':[param for name,param in model.named_parameters() if name in quantized_param_names], "lr":args.quant_lr},
            {'params':[param for name,param in model.named_parameters() if name not in quantized_param_names], "lr":args.learning_rate},
            {'params':distillation_transforms.parameters(), "lr":args.learning_rate},
        ],
        #model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    augmentations = Compose([
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    if args.dataset_name is not None:

        def transforms(examples):
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
            return {"input": images}

        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
        dataset.set_transform(transforms)
    else:
        df = pd.read_pickle(args.train_data_path)
        dataset = CustomDataset(df, augmentations)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )
    
    
    def lr_lambda(current_step: int):
        return max(0.998**current_step,1e-4)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    teacher = teacher.to(device)
    distillation_transforms = distillation_transforms.to(device)
    summary(teacher, [(1, 3, args.resolution, args.resolution), (1, )], verbose=1)

    loss_fn = F.l1_loss if args.use_l1_loss else F.mse_loss
    #scaler = torch.cuda.amp.GradScaler()
    global_step = 0
    losses = []
    avg_err = []
    avg_act_err = defaultdict(list)
    
    
    
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"].to(device)
            clean_images = normalize_to_neg_one_to_one(clean_images)

            batch_size = clean_images.shape[0]
            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(0,
                                      noise_scheduler.num_train_timesteps,
                                      (batch_size, ),
                                      device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            pred = model(noisy_images, timesteps)
            noise_pred = pred["sample"]
            if args.quantize:
                with torch.no_grad():
                    pred_teacher = teacher(noisy_images,timesteps)
                
                avg_err.append(F.l1_loss(noise_pred, pred_teacher["sample"]).detach().cpu().item())
                loss = 0
                
                if args.kd_layers == "skip":
                    pred["acts"] = [x for x,name in pred["acts"] if "skip" in name]
                    pred_teacher["acts"] = [x for x,name in pred_teacher["acts"] if "skip" in name]
                elif args.kd_layers == "up":
                    pred["acts"] = [x for x,name in pred["acts"] if "up" in name]
                    pred_teacher["acts"] = [x for x,name in pred_teacher["acts"] if "up" in name]
                else:
                    pred["acts"] = [x for x,name in pred["acts"]]
                    pred_teacher["acts"] = [x for x,name in pred_teacher["acts"]]
                for i,(a,b) in enumerate(zip(pred["acts"],pred_teacher["acts"])):
                    #print(f"{((a-b)**2).sum()/(b**2).sum()*100:.3f}",a.shape)
                    avg_act_err[f"{i}-{a.shape[1:]}"].append((((a-b)**2).sum()/(b**2).sum()*100).item())
                    features = a.shape[1]
                    a = a.transpose(1,3).reshape((-1,features))
                    b = b.transpose(1,3).reshape((-1,features))
                    
                    if args.use_dist_transform:
                        a = distillation_transforms[str(features)](a)
                    if args.kd_attention:
                        a = (a**2).sum(dim=1)
                        b = (b**2).sum(dim=1)
                    
                    
                    if args.hidden_dist==-1:
                        mul = 1.0/len(pred["acts"])
                    else:
                        mul = args.hidden_dist
                    loss += F.l1_loss(a, b)*mul
                loss += F.l1_loss(noise_pred, pred_teacher["sample"])*1
            else:
                loss = F.l1_loss(noise_pred, noise)
            loss.backward()
            
            

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "avg_err": numpy.mean(avg_err[-100:]),
                "step": global_step
            }

            progress_bar.set_postfix(**logs)
            if global_step % 500 == 0:
                with torch.no_grad():
                    # has to be instantiated every time, because of reproducibility
                    generator = torch.manual_seed(0)
                    generated_images = noise_scheduler.generate(
                        model,
                        num_inference_steps=n_inference_timesteps,
                        generator=generator,
                        eta=1.0,
                        use_clipped_model_output=True,
                        batch_size=args.eval_batch_size,
                        output_type="numpy")
                    
                    if args.quantize:
                        generator = torch.manual_seed(0)
                        generated_images_teacher = noise_scheduler.generate(
                            teacher,
                            num_inference_steps=n_inference_timesteps,
                            generator=generator,
                            eta=1.0,
                            use_clipped_model_output=True,
                            batch_size=args.eval_batch_size,
                            output_type="numpy")
                        torchvision.utils.save_image(torch.stack([generated_images["sample_pt"],generated_images_teacher["sample_pt"]]).transpose(0,1).reshape((-1,3,64,64)),
                            f"images/grid_{epoch}_{step}.png",
                            nrow=2)
                    else:
                        torchvision.utils.save_image(generated_images["sample_pt"],
                            "grid.png",
                            nrow=args.eval_batch_size // 4)
                
                # model_state_dict = model.state_dict()
                # for parameter_name, teacher_param in teacher.state_dict().items():
                #     if parameter_name not in quantized_param_names:
                #         print(f"{F.mse_loss(teacher_param,model_state_dict[parameter_name])*1000:.5f}",parameter_name)
                # for parameter_name, teacher_param in teacher.state_dict().items():
                #     if parameter_name in quantized_param_names:
                #         print(f"{F.mse_loss(teacher_param,model_state_dict[parameter_name])*1000:.5f}",parameter_name,math.prod(model_state_dict[parameter_name].shape),model_state_dict[parameter_name].shape)
                for name,errs in avg_act_err.items():
                    print(name,f"{numpy.mean(errs):.3f}")
                    
                torch.cuda.empty_cache()
            global_step += 1
            
        progress_bar.close()
        losses.append(losses_log / (step + 1))

        # Generate sample images for visual inspection
        if epoch % args.save_model_epochs == 0:
            with torch.no_grad():
                # has to be instantiated every time, because of reproducibility
                generator = torch.manual_seed(0)
                generated_images = noise_scheduler.generate(
                    model,
                    num_inference_steps=n_inference_timesteps,
                    generator=generator,
                    eta=1.0,
                    use_clipped_model_output=True,
                    batch_size=args.eval_batch_size,
                    output_type="numpy")

                save_images(generated_images, epoch, args)
                plot_losses(losses, f"{args.loss_logs_dir}/{epoch}/")

                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--dataset_name", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_path",
                        type=str,
                        default=None,
                        help="A df containing paths to training images.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="trained_models/ddpm-model-64.pth")
    parser.add_argument("--samples_dir", type=str, default="test_samples/")
    parser.add_argument("--loss_logs_dir", type=str, default="training_logs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--quant_lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=5e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--use_l1_loss", type=bool, default=False)
    parser.add_argument("--quantize",action='store_true')
    parser.add_argument("--qinit",action='store_true')
    parser.add_argument("--qres",action='store_true')
    parser.add_argument("--kd_attention",action='store_true')
    parser.add_argument("--hidden_dist", type=float, default=0.0)
    parser.add_argument("--use_dist_transform",action='store_true')
    parser.add_argument("--act_quant_bits", type=int, default=None)
    parser.add_argument("--kd_layers", type=str, default="")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="trained_models/ddpm-flowers.pth",
                        help="Path to pretrained model")

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_path is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)