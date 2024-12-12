# Video-to-Audio

## Description

Video to Audio pipeline consists of a visual encoder, a text encoder, UNet diffusion model to generate spectrogram and Griffin-lim algorithm to convert spectrogram into audio. 
Visual and text encoders share the same multimodal visual language decoder ([cogvlm2-video-llama3-chat](link)). 

Our UNet diffusion model is a finetune of the music generation model [riffusion](https://huggingface.co/riffusion/riffusion-model-v1). We made modifications in the architecture to condition on video frames and improve the synchronization between video and audio. Also, we replace the text encoder with the decoder of [cogvlm2-video-llama3-chat](link).

![pipeline-audio](https://github.com/user-attachments/assets/f5d6fafb-6e0a-4362-b6e0-63c51c79bfc2)

## Installation

```bash
git clone https://github.com/ai-forever/Kandinsky-4.git
cd Kandinsky-4
conda install -c conda-forge ffmpeg -y
pip install -r kandinsky4_video2audio/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
```

## Inference

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

