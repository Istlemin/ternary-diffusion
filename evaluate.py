import argparse

import torch
import torch.nn.functional as F

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from datasets import load_dataset
from scheduler import DDIMScheduler
from model import UNet
from tqdm import tqdm
from quantization import calc_model_size, quantize_unet, QuantizedConv2d, QuantizedLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_timesteps = 1000
n_inference_timesteps = 50

def eval(model,args):
    augmentations = Compose([
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution),
        ToTensor(),
    ])
    
     
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

    dataset_size = args.dataset_size or len(dataset)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size, shuffle=False)

    batch = next(iter(train_dataloader))
    all_dataset_images = batch["input"]
    
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")
    
    with torch.no_grad():
        # has to be instantiated every time, because of reproducibility
        generator = torch.manual_seed(0)
        all_generated_images = []
        for i in tqdm(range((dataset_size+args.eval_batch_size-1)//args.eval_batch_size)):
            generated_images = noise_scheduler.generate(
                model,
                num_inference_steps=n_inference_timesteps,
                generator=generator,
                eta=1.0,
                use_clipped_model_output=True,
                batch_size=args.eval_batch_size,
                output_type="numpy")

            all_generated_images.append(generated_images["sample_pt"].cpu())
        
        all_generated_images = torch.cat(all_generated_images)[:dataset_size]

    print(len(all_dataset_images))
    print(len(all_generated_images))
    
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(normalize=True)
    fid.update(all_dataset_images, real=True)
    fid.update(all_generated_images, real=False)

    print(f"FID: {float(fid.compute())}")

def main(args):
    model = UNet(3, image_size=args.resolution, hidden_dims=[16, 32, 64, 128])

    pretrained = torch.load(args.pretrained_model_path)["model_state"]
    model.load_state_dict(pretrained, strict=False)

    model.load_state_dict(pretrained, strict=False)

    model, quantized_param_names = quantize_unet(model,args)
    
    model.load_state_dict(torch.load(args.pretrained_model_path)["model_state"])

    model = model.to(device)

    eval(model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple script for image generation.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--quantize",action='store_true')
    parser.add_argument("--qinit",action='store_true')
    parser.add_argument("--qres",action='store_true')
    parser.add_argument("--quant_emb",action='store_true')
    parser.add_argument("--kd_attention",action='store_true')
    parser.add_argument("--hidden_dist", type=float, default=0.0)
    parser.add_argument("--use_dist_transform",action='store_true')
    parser.add_argument("--act_quant_bits", type=int, default=None)
    parser.add_argument("--kd_layers", type=str, default="")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Path to pretrained model")

    args = parser.parse_args()

    main(args)