# Kandinsky 4.0: text-to-video, image-to-video, and video-to-audio diffusion models
<br><br><br><br>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/KANDINSKY_LOGO_1_WHITE.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/KANDINSKY_LOGO_1_BLACK.png">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  </picture>
</div>

<div align="center">
  <a>Kandinsky 4.0 Post</a> | <a href=https://ai-forever.github.io/Kandinsky-4/K40/>Project Page</a> | <a href=https://huggingface.co/spaces/ai-forever/kandinsky-4-t2v-flash>Generate</a> | <a>Technical Report</a> | <a href=https://github.com/ai-forever/Kandinsky-4>GitHub</a> | <a href=https://huggingface.co/ai-forever/kandinsky-4-t2v-flash> Kandinsky 4.0 T2V Flash HuggingFace</a> | <a href=https://huggingface.co/ai-forever/kandinsky-4-v2a> Kandinsky 4.0 V2A HuggingFace</a>
</div>

<div align="center">
  This repository is the official implementation of Kandinsky 4.0 T2V Flash and Kandinsky 4.0 V2A.
</div>

<br><br><br><br>


## Kandinsky 4.0 T2V (text-to-video)

**Open source: SOON** 🤗

### Examples:

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/b324ee89-3deb-49e7-a506-92f67cd7dc54" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fc7eb771-dd5e-4a5c-815a-6657f44b8e22" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/300af939-7516-4521-8eeb-6a660b70fb85" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/13a64f44-8d81-4760-8a69-dcd483ee62f2" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fe99f74d-7ea9-41e1-9bb4-45ab258eb254" width=200 controls autoplay loop></video>
      </td>
  </tr>

</table>

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/9bc737bc-9a58-46a1-b4d0-95df0a34ff88" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/c80362fd-e173-44c2-a774-6f36810be442" width=200 controls autoplay loop></video>
      </td>
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/98012d67-5aa3-492a-82d9-9cb72f3c3cf8" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/c1e11759-3e15-415d-b9e8-4995b44e9194" width=200 controls autoplay loop></video>
      </td>
  </tr>

</table>

## Kandinsky 4.0 T2V Flash (distilled version of Kandinsky 4.0 T2V 480p)

### Examples:

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/c52f5734-79de-4d09-b738-0ad09ffda25f" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/a0218156-3bc2-4464-8c06-84aac8c75927" width=200 controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/91c6ee64-84a7-45a4-9e08-7eceacb8518a" width=200 controls autoplay loop></video>
      </td>
  </tr>

</table>

### Description:

Kandinsky 4.0 is a text-to-video generation model based on latent diffusion for 480p and HD resolutions. Here we present distiled version of this model **Kandisnly 4 flash**, that can generate **12 second videos** in 480p resolution in **11 seconds** on a single NVIDIA H100 gpu. The pipeline consist of 3D causal [CogVideoX](https://arxiv.org/pdf/2408.06072) VAE, text embedder [T5-V1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl) and our trained MMDiT-like transformer model.

<img src="assets/pipeline.png" height=500>

A serious problem for all diffusion models, and especially video generation models, is the generation speed. To solve this problem, we used the Latent Adversarial Diffusion Distillation (LADD) approach, proposed for distilling image generation models and first described in the [article](https://arxiv.org/pdf/2403.12015) from Stability AI and tested by us when training the [Kandinsky 3.1](https://github.com/ai-forever/Kandinsky-3) image generation model. The distillation pipeline itself involves additional training of the diffusion model in the GAN pipeline, i.e. joint training of the diffusion generator with the discriminator.

<img src="assets/LADD.png">


### Architecture

For training Kandinsky 4 Flash we used the following architecture of diffusion transformer, based on MMDiT proposed in [Stable Diffusion 3](https://arxiv.org/pdf/2403.03206).

<img src="assets/MMDiT1.png" width="50%"> <img src="assets/MMDiT_block1.png" width="30%">

For training flash version we used the following architecture of discriminator. Discriminator head structure resembles half of an MMDiT block.

<img src="assets/discriminator.png" width="50%"> <img src="assets/discriminator_head.png" width="30%">


### How to use Kandinsky 4.0 T2V Flash:

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

## Kandinsky 4.0 I2V (image-to-video)

**Open source: SOON** 🤗

### Examples:

<table border="0" style="width: 100; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/generation_examples/I2V/IMG_9680.jpg">
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e17f4384-9f81-4a75-a3b8-fe2431c30fac" width=200 controls autoplay loop></video>
      </td>
  </tr>
  <table border="0" style="width: 100; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/generation_examples/I2V/IMG_9677.jpg">
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/2116ace3-3edf-434d-a425-b13309a35bb7" width=200 controls autoplay loop></video>
      </td>
  </tr>
  <table border="0" style="width: 100; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="assets/generation_examples/I2V/IMG_9675.jpg">
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/dbfa54ac-225d-4d30-94c1-eb371a5c6031" width=200 controls autoplay loop></video>
      </td>
  </tr>

</table>

## Kandinsky 4.0 V2A (video-to-audio)

### Examples:

<table border="0" style="width: 200; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/6bb5cb9c-00b4-4d7a-9616-a1debf456e02" width=200 controls autoplay loop playsinline></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/1eb223af-c743-4948-9532-9e6e097b979a" width=200 controls autoplay loop playsinline></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/cf22eeee-67aa-4b32-bea8-23d6954852a5" width=200 controls autoplay loop playsinline></video>
      </td>
  </tr>
</table>


### Description:
We also release generation pipeline able to produce audio for Kandinsky 4.0 video outputs. 
![pipeline-audio](https://github.com/user-attachments/assets/f5d6fafb-6e0a-4362-b6e0-63c51c79bfc2)

Video to Audio pipeline consists of a visual encoder, a text encoder, UNet diffusion model to generate spectrogram and Griffin-lim algorithm to convert spectrogram into audio. 
Visual and text encoders share the same multimodal visual language decoder ([cogvlm2-video-llama3-chat](link)). 

Our UNet diffusion model is a finetune of the music generation model [riffusion](https://huggingface.co/riffusion/riffusion-model-v1). We made modifications in the architecture to condition on video frames and improve the synchronization between video and audio. Also, we replace the text encoder with the decoder of [cogvlm2-video-llama3-chat](link).


Inference code for Video-to-Audio:

```python
import torch
import torchvision

from kandinsky4_video2audio.video2audio_pipe import Video2AudioPipeline
from kandinsky4_video2audio.utils import load_video, create_video

device='cuda:0'

pipe = Video2AudioPipeline(
    "ai-forever/kandinsky-4-v2a",
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


# Authors

### Project Leader 

Denis Dimitrov

### Scientific Consultant

Andrey Kuznetsov, Sergey Markov

### Training Pipeline & Model Pretrain & Model Distillation

Vladimir Arkhipkin, Novitskiy Lev, Maria Kovaleva

### Model Architecture

Vladimir Arkhipkin, Maria Kovaleva, Zein Shaheen, Arsen Kuzhamuratov, Nikolay Gerasimenko, Mikhail Zhirnov, Alexandr Gambashidze, Konstantin Sobolev

### Data Pipeline

Ivan Kirillov, Andrei Shutkin, Kirill Chernishev, Julia Agafonova, Denis Parkhomenko

### Video-to-audio model

Zein Shaheen, Arseniy Shakhmatov, Denis Parkhomenko

### Quality Assessment

Nikolay Gerasimenko, Anna Averchenkova

### Other Contributors

Viacheslav Vasilev, Andrei Filatov, Gregory Leleytner
