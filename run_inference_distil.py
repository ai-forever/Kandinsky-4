import torch
from kandinsky import get_T2V_pipeline
from argparse import ArgumentParser
import time


if __name__ == "__main__":    
    device_map = {
        "dit": torch.device('cuda:0'), 
        "vae": torch.device('cuda:0'), 
        "text_embedder": torch.device('cuda:0')
    }
    pipe = get_T2V_pipeline(device_map, resolution=512)

    prompt = "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance"

    pipe(
        text=prompt,
        time_length=12,
        width=672,
        height=384,
        seed=42,
        save_path="./test.mp4",
    )