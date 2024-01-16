# Ternary Diffusion model

An implementation of ternarisation for a diffusion model, based on filipbasara0's [simple-diffusion](https://github.com/filipbasara0/simple-diffusion) repository. Modifications have been made to train.py, model/unet.py and generate.py, and files quantization.py and evaluate.py have been added.

To train the best perforing ternarised model, run the following command:
```
python train.py --num_epochs 5 --learning_rate=1e-2 --quant_lr=1e-1 --quantize --quant_emb --lr_schedule exponential --hidden_dist=-1 --kd_layers all --output_dir "trained_models/ternarised_model"
```
