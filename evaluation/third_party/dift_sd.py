# code credit: https://github.com/Tsingularity/dift/blob/main/src/models/dift_sd.py
# Adapted from DragLoRA's official drag_bench_evaluation/dift_sd.py.
import gc
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
try:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel
except ModuleNotFoundError:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        default_overall_up_factor = 2**self.num_upsamplers

        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type == "text_time":
            if added_cond_kwargs is None:
                raise ValueError("added_cond_kwargs must be provided for text_time addition embeddings")
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            if text_embeds is None or time_ids is None:
                raise ValueError("text_embeds and time_ids are required in added_cond_kwargs")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            emb = emb + self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "text":
            emb = emb + self.add_embedding(encoder_hidden_states)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        return {"up_ft": up_ft}


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        timestep = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, timestep)
        return self.unet(
            latents_noisy,
            timestep,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )


class SDFeaturizer:
    def __init__(self, sd_id, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to(self.device)
        onestep_pipe.enable_attention_slicing()
        self.pipe = onestep_pipe

    @torch.no_grad()
    def forward(self, img_tensor, prompt, t=261, up_ft_index=1, ensemble_size=8):
        """
        DragLoRA/DIFT-style Stable Diffusion feature extraction.

        Args:
            img_tensor: a tensor in [1, C, H, W] or [C, H, W], normalized to [-1, 1].
            prompt: text prompt used for Stable Diffusion conditioning.
            t: diffusion timestep.
            up_ft_index: U-Net upsampling block index.
            ensemble_size: repeated noisy samples averaged for features.
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).to(self.device)
        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
        )
        unet_ft = unet_ft_all["up_ft"][up_ft_index]
        return unet_ft.mean(0, keepdim=True)
