# Kandinsky4-Audio: Video-to-Audio Generation

## Pipeline

<p align="center">
<img src="__assets__/pipeline.png" width="800px"/>
<br>
<em>In the <a href="https://github.com/ai-forever/KandinskyVideo/tree/kandinsky_video_1_0">Kandinsky Video 1.0</a>, the encoded text prompt enters the text-to-video U-Net3D keyframe generation model with temporal layers or blocks, and then the sampled latent keyframes are sent to the latent interpolation model to predict three interpolation frames between
two keyframes. An image MoVQ-GAN decoder is used to obtain the final video result. In <B>Kandinsky Video 1.1</B>, text-to-video U-Net3D is also conditioned on text-to-image U-Net2D, which helps to improve the content quality. A temporal MoVQ-GAN decoder is used to decode the final video.</em>
</p>


# Installation

# Run

# Samples