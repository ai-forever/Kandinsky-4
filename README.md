# Kandinsky-4 flash: Text-to-Video diffusion model

![]()

[Kandinsky 4.0 Post]() | [Project Page]() | [Generate]() | [Telegram-bot]() | [Technical Report]() | [GitHub](https://github.com/ai-forever/Kandinsky-4) | [HuggingFace](https://huggingface.co/ai-forever/kandinsky4) |

## Description:

Kandinsky 4.0 is a text-to-video generation model based on latent diffusion for 480p and HD resolutions. Here we present distiled version of this model Kandisnly 4 flash, that can generate 12 second videos in 480p resolution in 11 seconds on single gpu. The pipeline consist of 3D causal [CogVideoX](https://arxiv.org/pdf/2408.06072) VAE, text embedder [T5-V1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl) and our trained MMDiT-like transformer model.

<img src="assets/pipeline.png">

A serious problem for all diffusion models, and especially video generation models, is the generation speed. To solve this problem, we used the Latent Adversarial Diffusion Distillation (LADD) approach, proposed for distilling image generation models and first described in the [article](https://arxiv.org/pdf/2403.12015) from Stability AI and tested by us when training the [Kandinsky 3.1](https://github.com/ai-forever/Kandinsky-3) image generation model. The distillation pipeline itself involves additional training of the diffusion model in the GAN pipeline, i.e. joint training of the diffusion generator with the discriminator.

<img src="assets/LADD.png">


## Architecture

For training Kandinsky 4 Flash we used the following architecture of diffusion transformer, based on MMDiT proposed in [Stable Diffusion 3](https://arxiv.org/pdf/2403.03206).

<img src="assets/MMDiT1.png"> <img src="assets/MMDiT_block1.png">

For training flash version we used the following architecture of discriminator. Discriminator head structure resembles half of an MMDiT block.

<img src="assets/discriminator.png"> <img src="assets/discriminator_head.png">


## How to use:

### 1. Text-to-Video
```python
import torch
from IPython.display import Video
from kandinsky import get_T2V_pipeline

device_map = {
    "dit": torch.device('cuda:0'), 
    "vae": torch.device('cuda:0'), 
    "text_embedder": torch.device('cuda:0')
}

pipe = get_T2V_pipeline(device_map)

images = pipe(
    seed=42,
    time_length=12,
    width = 672,
    height = 384,
    save_path="./test.mp4",
    text="Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance",
)

Video("./test.mp4")
```

Examples of usage and more detailed parameters description are in the [examples.ipynb](examples.ipynb) notebook.

Make sure that you have weights folder with weights of all models.

We also add distributed inference opportunity: [run_inference_distil.py](run_inference_distil.py)

To run this examples:
```
python -m torch.distributed.launch --nnodes n --nproc-per-node m run_inference_distil.py
```
where n is a number of nodes you have and m is a number of gpus on these nodes. The code was tested with n=1 and m=8, so this is preferable parameters.

In distributed setting the DiT models are parallelized using tensor parallel on all gpus, which enables a significant speedup.

To run this examples from terminal without tensor parallel:
```
python run_inference_distil.py
```

### 2. Video-to-Audio

```python
import torch
import torchvision

from kandinsky4_video2audio.video2audio_pipe import Video2AudioPipeline
from kandinsky4_video2audio.utils import load_video, create_video

device='cuda:0'

pipe = Video2AudioPipeline(
    "ai-forever/kandinsky4-Audio",
    torch_dtype=torch.float16,
    device = device
)

video_path = 'assets/inputs/1.mp4'
video, _, fps = torchvision.io.read_video(video_path)

prompt="clean. clear. good quality."
negative_prompt = "hissing noise. drumming rythm. saying. poor quality."
video_input, video_complete, duration_sec = load_video(video, fps['video_fps'], num_frames=96, max_duration_sec=12)
    
out = pipe(
    video_input,
    prompt,
    negative_prompt=negative_prompt,
    duration_sec=duration_sec, 
)[0]

save_path = f'assets/outputs/1.mp4'
create_video(
    out, 
    video_complete, 
    display_video=True,
    save_path=save_path,
    device=device
)
```

https://github.com/user-attachments/assets/49756e68-4e80-41a0-a09b-d886d7e0d6e7


# Authors
+ Lev Novitkiy: [GitHub](https://github.com/leffff), [Blog](https://t.me/mlball_days)
+ Maria Kovaleva [GitHub](https://github.com/MarKovka20)
+ Vladimir Arkhipkin: [GitHub](https://github.com/oriBetelgeuse)
+ Denis Parkhomenko: [GitHub](https://github.com/nihao88)
+ Andrei Shutkin: [GitHub](https://github.com/maleficxp)
+ Ivan Kirillov: [GitHub](https://github.com/funnylittleman)
+ Zein Shaheen: [GitHub](https://github.com/zeinsh)
+ Viacheslav Vasilev: [GitHub](https://github.com/vivasilev)
+ Andrei Filatov [GitHub](https://github.com/anvilarth)
+ Julia Agafonova
+ Nikolay Gerasimenko [GitHub](https://github.com/Nikolay-Gerasimenko)
+ Andrey Kuznetsov: [GitHub](https://github.com/kuznetsoffandrey), [Blog](https://t.me/complete_ai)
+ Denis Dimitrov: [GitHub](https://github.com/denndimitrov), [Blog](https://t.me/dendi_math_ai)
