from typing import Any, Callable, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from kandinsky4_video2audio.model.unet import UNet2DConditionModel

class Video2AudioPipeline:

    def __init__(
        self,
        path_to_model,
        torch_dtype = torch.float16,
        device = 'cuda'
    ):
        super().__init__()
        
        self.scheduler = PNDMScheduler.from_pretrained(path_to_model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(path_to_model, subfolder="vae", torch_dtype=torch.float16).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/cogvlm2-video-llama3-chat",
            trust_remote_code=True,
        )
        
        self.multimodal = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm2-video-llama3-chat",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            path_to_model, 
            subfolder='unet', 
            torch_dtype=torch.bfloat16,
        ).to(device)

        self._execution_device = device
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
        
    def __prepare_inputs__(self, images=[], query=""):
        if type(images)!=list:
            images=[images]
            
        inputs = self.multimodal.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            images=images,
            history=[],
            template_version="base"
        )
    
        if len(images)>0:
            images = inputs['images']=[[inputs['images'][0].to(self.unet.device).to(self.multimodal.dtype)]]
            
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.unet.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.unet.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.unet.device),
        }

        if len(images)>0:
            inputs['images'] = images
            
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": True,
            "top_p": 0.1,
            "temperature": 0.1,
        }
        return inputs, gen_kwargs        

    def extract_video_embedding(self, images):
        inputs, gen_kwargs = self.__prepare_inputs__(images=images)

        emb = self.multimodal(**inputs, output_hidden_states=True)
        emb=emb.hidden_states[-2][0, inputs['token_type_ids'][0]==1]
        emb=emb.reshape((images.shape[1], -1, emb.shape[-1])).mean(dim=1)
    
        return emb

    def extract_text_embedding(self, text, max_len=100):
        inputs, gen_kwargs = self.__prepare_inputs__(query=text)

        emb = self.multimodal(**inputs, output_hidden_states=True)
        emb=emb.hidden_states[-2][0, inputs['token_type_ids'][0]==0]

        pad_len=max_len-emb.shape[0]
        if pad_len>0:
            emb=torch.cat([emb, torch.zeros(pad_len, emb.shape[1], device=emb.device)], dim=0)

        return emb[:max_len]

    def _encode_conditions(
        self,
        prompt,
        images,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        prompt_embeds_dtype = self.unet.dtype
        prompt_embeds = self.extract_text_embedding(prompt).unsqueeze(0)
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        
        image_embeds = self.extract_video_embedding(images).unsqueeze(0)
        image_embeds = image_embeds.to(dtype=prompt_embeds_dtype, device=device)
        
        if do_classifier_free_guidance:
            negative_prompt_embeds = self.extract_text_embedding(negative_prompt).unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            uncond_image_embeds = self.extract_video_embedding(images*0).unsqueeze(0)
            uncond_image_embeds = uncond_image_embeds.to(dtype=prompt_embeds_dtype, device=device)
            image_embeds = torch.cat([uncond_image_embeds, image_embeds])

        return prompt_embeds, image_embeds

    @torch.no_grad()
    def __call__(
        self,
        images: torch.FloatTensor,
        prompt: str,
        negative_prompt: str = None,
        height: int = 512,
        duration_sec: int = 10,
        num_inference_steps: int = 50,
        guidance_scale: float = 8,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
    ):
        """
        Generate an audio spectrogram from input video frames and textual prompts using a generative model.

        Args:
            images (torch.FloatTensor): Input tensor representing the frames used for generation.
            prompt (str): Textual description that guides the generative model.
            negative_prompt (Optional[Union[str, List[str]]]): Text descriptions to guide the model away 
                from specific concepts.
            height (int): Height of the generated output. Default is 512.
            duration_sec (int): Duration in seconds for audio output. Default is 10 seconds.
            num_inference_steps (int): Number of inference steps for the generative process. Higher values 
                increase quality but also computation time. Default is 50.
            guidance_scale (float): Scale factor for prompt guidance. Higher values increase adherence 
                to the prompt at the cost of creativity. Default is 8.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]]): Random number generator(s) 
                for reproducibility. Can be a single generator or a list of generators for multiple seeds.
            latents (Optional[torch.FloatTensor]): Precomputed latent codes to initialize the generation process.
            output_type (Optional[str]): Format of the output. Options are "pil" (PIL.Image) or other custom types. 
                Default is "pil".
        Returns:
            returns the spectrogram tensor.
        """
        batch_size = 1
        width = ((duration_sec*100) // 32) * 32
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, image_embeds = self._encode_conditions(
            prompt,
            images,
            device,
            do_classifier_free_guidance,
            negative_prompt,
        )

        cross_attention_kwargs = {"image_embeds": image_embeds}
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        for i, t in tqdm(enumerate(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        spectogram = self.vae.decode(latents.to(self.vae.dtype) / self.vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * spectogram.shape[0]
        spectogram = self.image_processor.postprocess(spectogram, output_type=output_type, do_denormalize=do_denormalize)

        return spectogram