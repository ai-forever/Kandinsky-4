import os
from typing import Optional, Union

import torch
from omegaconf import OmegaConf
from .model.dit import get_dit, parallelize
from .model.text_embedders import get_text_embedder
from diffusers import AutoencoderKLCogVideoX
from omegaconf.dictconfig import DictConfig
# from huggingface_hub import hf_hub_download, snapshot_download

from .t2v_pipeline import Kandinsky4T2VPipeline

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from diffusers import FluxPipeline

def get_T2V_pipeline(
        device_map: Union[str, torch.device, dict],
        resolution: int = 512,
        cache_dir: str = './weights/',
        dit_path: str = None,
        text_embedder_path: str = None,
        vae_path: str = None,
        scheduler_path: str = None,
        conf_path: str = None,
) -> Kandinsky4T2VPipeline:
    
    assert resolution in [512]
    
    if not isinstance(device_map, dict):
        device_map = {
            'dit': device_map, 
            'vae': device_map, 
            'text_embedder': device_map
        }

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    except:
        local_rank, world_size = 0, 1
        
    if world_size > 1:
        device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tensor_parallel",))
        device_map["dit"] = torch.device(f'cuda:{local_rank}')

    #TODO hf_hub_download, snapshot_download
    if dit_path is None:
        dit_path = os.path.join(cache_dir, f"kandinsky4_distil_{resolution}.pt")
            
    if text_embedder_path is None:
        text_embedder_path = os.path.join(cache_dir, f"t5_v1_1_xxl/")
    if vae_path is None:
        vae_path = os.path.join(cache_dir, f"vae/")
    if scheduler_path is None:
        scheduler_path = os.path.join(cache_dir, f"scheduler_{resolution}/")

    if conf_path is None:
        conf = get_default_conf(vae_path, text_embedder_path, scheduler_path, dit_path)
    else:
        conf = OmegaConf.load(conf_path)

    noise_scheduler, dit = get_dit(conf.dit)
    dit = dit.to(dtype=torch.bfloat16, device=device_map["dit"])
    
    if world_size > 1:
        dit = parallelize(dit, device_mesh["tensor_parallel"])
        
    text_embedder = get_text_embedder(conf)
    text_embedder = text_embedder.freeze()
    if local_rank == 0:
        text_embedder = text_embedder.to(device=device_map["text_embedder"], dtype=torch.bfloat16)
    
    vae = AutoencoderKLCogVideoX.from_pretrained(conf.vae.checkpoint_path)
    vae = vae.eval()
    if local_rank == 0:
        vae = vae.to(device_map["vae"], dtype=torch.bfloat16)

    return Kandinsky4T2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        resolution=resolution,
        local_dit_rank=local_rank,
        world_size=world_size,
    )


def get_default_conf(
    vae_path,
    text_embedder_path, 
    scheduler_path, 
    dit_path, 
) -> DictConfig:
    dit_params = {
            'in_visual_dim': 16, 
            'in_text_dim': 4096, 
            'out_visual_dim': 16, 
            'time_dim': 512, 
            'patch_size': [1, 2, 2], 
            'model_dim': 3072, 
            'ff_dim': 12288, 
            'num_blocks': 21, 
            'axes_dims': [16, 24, 24]
        }
    
    conf = {
        'vae': 
            {
                'checkpoint_path': vae_path
            }, 
        'text_embedder': 
            {
                'emb_size': 4096, 
                'tokens_lenght': 224, 
                'params': 
                    {
                        'checkpoint_path': text_embedder_path
                    }
            }, 
        'dit': 
            {
                'scheduler': scheduler_path, 
                'checkpoint_path': dit_path, 
                'params': dit_params
                
            }, 
        'resolution': 512, 
    }
    
    return DictConfig(conf)