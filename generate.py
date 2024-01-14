import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

import os
from PIL import Image
from torchvision import utils

from scheduler import DDIMScheduler
from model import UNet
from quantization import calc_model_size, quantize_unet, QuantizedConv2d, QuantizedLinear

n_timesteps = 1000
n_inference_timesteps = 50


def main(args):
    model = UNet(3, image_size=args.resolution, hidden_dims=[16, 32, 64, 128])
    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")

    pretrained = torch.load(args.pretrained_model_path)["model_state"]
    model.load_state_dict(pretrained, strict=False)

    model.load_state_dict(pretrained, strict=False)

    model, quantized_param_names = quantize_unet(model,args)
    
    model.load_state_dict(torch.load(args.pretrained_model_path)["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        # has to be instantiated every time, because of reproducibility
        generator = torch.manual_seed(0)
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=n_inference_timesteps,
            generator=generator,
            eta=0.5,
            use_clipped_model_output=True,
            batch_size=args.eval_batch_size,
            output_type="numpy")

        images = generated_images["sample"]
        images_processed = (images * 255).round().astype("uint8")

        current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
        out_dir = f"./{args.samples_dir}/{current_date}/"
        os.makedirs(out_dir)
        for idx, image in enumerate(images_processed):
            image = Image.fromarray(image)
            image.save(f"{out_dir}/{idx}.jpeg")

        utils.save_image(generated_images["sample_pt"],
                         f"{out_dir}/grid.jpeg",
                         nrow=args.eval_batch_size // 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple script for image generation.")
    parser.add_argument("--samples_dir", type=str, default="test_samples/")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=4)
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