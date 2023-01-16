"""
SAR-DDPM Inference showoff.
"""

import argparse
import torch
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

from guideddiffusion import dist_util, logger
from guideddiffusion.image_datasets import load_data
from guideddiffusion.resample import create_named_schedule_sampler
from guideddiffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guideddiffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from torch.optim import AdamW

from valdata import  ValData, ValDataNew, ValDataNewReal
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

val_dir = '../DSIFN/smallTest/'
base_path = '../testResults/'
resume_checkpoint_clean = './weights/sar_ddpm.pt'




def main():
    args = create_argparser().parse_args()
    
    model_clean, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    
    


    val_data = DataLoader(ValDataNew(dataset_path=val_dir), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()

    device0 = torch.device("cpu")
    
    model_clean.load_state_dict(torch.load(resume_checkpoint_clean, map_location="cpu"))

    
    model_clean.to(device0)

    
    
    
    print('model clean device:')
    print(next(model_clean.parameters()).device)

    

    with torch.no_grad(): 
        number = 0
        

        for batch_id1, data_var in enumerate(val_data):
            number = number+1 
            clean_batch, model_kwargs1 = data_var

            speck = model_kwargs1['SR']
            speck = ((speck + 1) * 127.5)
            speck = speck.clamp(0, 255).to(torch.uint8)
            speck = speck.permute(0, 2, 3, 1)
            speck = speck.contiguous().cpu().numpy()
            speck = speck[0][:,:,::-1]

            plt.imsave(base_path+'speck_'+str(batch_id1)+".png", speck)


            sample_new = diffusion.p_sample_loop(
                            model_clean,
                            (clean_batch.shape[0], 3, 256,256),
                            clip_denoised=True,
                            model_kwargs=model_kwargs1,
                        )
            
            sample_new = ((sample_new + 1) * 127.5)
            sample_new = sample_new.clamp(0, 255).to(torch.uint8)
            sample_new = sample_new.permute(0, 2, 3, 1)
            sample_new = sample_new.contiguous().cpu().numpy()
            sample_new = sample_new[0][:,:,::-1]
            
            sample_new = cv2.cvtColor(sample_new, cv2.COLOR_BGR2GRAY)
            print(batch_id1)
            plt.imsave(base_path+'pred_'+str(batch_id1)+".png", sample_new)

                




def create_argparser():
    defaults = dict(
        data_dir= val_dir,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=200,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
