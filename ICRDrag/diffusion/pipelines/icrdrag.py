import einops
import inspect
import torch
import numpy as np
import PIL
import copy
from dataclasses import dataclass
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import (
    BaseOutput,
    is_accelerate_available,
    is_accelerate_version,
    is_torch_version,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import T5EncoderModel, T5Tokenizer
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

from ICRDrag.models.denoiser.nextdit import NextDiT
from ICRDrag.dataset.utils import *
from ICRDrag.diffusion.pipelines.image_processor import VaeImageProcessorICRDrag

try:
    from ICRDrag.dataset.multitask.multiview import calculate_rays
except ImportError:
    calculate_rays = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SUPPORTED_DEVICE_MAP = ["balanced"]

def create_c2w_matrix(azimuth_deg, elevation_deg, distance=1.0, target=np.array([0, 0, 0])):
    """
    Create a Camera-to-World (C2W) matrix from azimuth and elevation angles.

    Parameters:
    - azimuth_deg: Azimuth angle in degrees.
    - elevation_deg: Elevation angle in degrees.
    - distance: Distance from the target point.
    - target: The point the camera is looking at in world coordinates.

    Returns:
    - C2W: A 4x4 NumPy array representing the Camera-to-World transformation matrix.
    """
    # Convert angles from degrees to radians
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)

    # Spherical to Cartesian conversion for camera position
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    camera_position = np.array([x, y, z])

    # Define the forward vector (from camera to target)
    target = 2*camera_position - target
    forward = target - camera_position
    forward /= np.linalg.norm(forward)

    # Define the world up vector
    world_up = np.array([0, 0, 1])

    # Compute the right vector
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        # Handle the singularity when forward is parallel to world_up
        world_up = np.array([0, 1, 0])
        right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)

    # Recompute the orthogonal up vector
    up = np.cross(forward, right)

    # Construct the rotation matrix
    rotation = np.vstack([right, up, forward]).T  # 3x3

    # Construct the full C2W matrix
    C2W = np.eye(4)
    C2W[:3, :3] = rotation
    C2W[:3, 3] = camera_position

    return C2W

@dataclass
class ICRDragPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[Image.Image], np.ndarray]
    latents: Optional[torch.Tensor] = None


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    
    
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
    # max_clip: float = 1.5,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len) # 0.000169270833
    b = base_shift - m * base_seq_len # 0.5-0.0433333332
    mu = image_seq_len * m + b
    # mu = min(mu, max_clip)
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class ICRDragPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using ICRDrag.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        transformer ([`NextDiT`]):
            Conditional transformer (NextDiT) architecture to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. ICRDrag uses the T5 model as text encoder.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class T5Tokenizer.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    def __init__(
        self,
        transformer: NextDiT,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        self.copy_scheduler = copy.deepcopy(scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessorICRDrag(vae_scale_factor=self.vae_scale_factor)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.transformer, "_hf_hook"):
            return self.device
        for module in self.transformer.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        max_length=300,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )

        text_encoder_output = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask.to(device))
        prompt_embeds = text_encoder_output[0].to(torch.float32)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_images_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_images_per_prompt, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_encoder_output = self.text_encoder(uncond_input.input_ids.to(device), attention_mask=uncond_input.attention_mask.to(device))
            negative_prompt_embeds = uncond_encoder_output[0].to(torch.float32)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # duplicate unconditional attention mask for each generation per prompt
            uncond_attention_mask = uncond_input.attention_mask.repeat(1, num_images_per_prompt)
            uncond_attention_mask = uncond_attention_mask.view(batch_size * num_images_per_prompt, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([uncond_attention_mask, attention_mask])

        return prompt_embeds.to(device), attention_mask.to(device)

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        denoise_mask: Optional[List[int]] = [1, 0],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        do_crop: bool = True,
        is_multiview: bool = False,
        multiview_azimuths: Optional[List[int]] = [0, 30, 60, 90],
        multiview_elevations: Optional[List[int]] = [0, 0, 0, 0],
        multiview_distances: float = 1.7,
        multiview_c2ws: Optional[List[torch.Tensor]] = None,
        multiview_intrinsics: Optional[torch.Tensor] = None,
        multiview_focal_length: float = 1.3887,
        forward_kwargs: Optional[Dict[str, Any]] = {},
        noise_scale: float = 1.0,
        **kwargs,
):
        # Convert single image to list for consistent handling
        if isinstance(image, PIL.Image.Image):
            image = [image]
            
        if height is None or width is None:
            closest_ar = get_closest_ratio(height=image[0].size[1], width=image[0].size[0], ratios=ASPECT_RATIO_512)
            height, width = int(closest_ar[0][0]), int(closest_ar[0][1])
        
        if not isinstance(multiview_distances, list) and not isinstance(multiview_distances, tuple):
            multiview_distances = [multiview_distances] * len(multiview_azimuths)
            
        # height = height or self.transformer.config.input_size[-2] * 8  # VAE downscale factor.
        # width = width or self.transformer.config.input_size[-1] * 8

        # 1. check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Additional input validation for image list
        if not all(isinstance(img, PIL.Image.Image) for img in image):
            raise ValueError("All elements in image list must be PIL.Image objects")

        # 2. define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        encoder_hidden_states, encoder_attention_mask = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        # 4. Preprocess all images
        if image is not None and len(image) > 0:
            processed_image = self.image_processor.preprocess(image, height=height, width=width, do_crop=do_crop)
        else:
            processed_image = None
            
        # # Stack processed images along the sequence dimension
        # if len(processed_images) > 1:
        #     processed_image = torch.cat(processed_images, dim=0)
        # else:
        #     processed_image = processed_images[0]

        timesteps = None

        # 6. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        if processed_image is not None:
            cond_latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                self.dtype,
                device,
                generator,
                latents,
                image=processed_image,
            )
            noise_pixel = torch.randn_like(cond_latents)
            cond_latents = self.copy_scheduler.sigmas[-1] * noise_pixel + (1 - self.copy_scheduler.sigmas[-1]) * cond_latents
        else:
            cond_latents = None
            
        # 7. prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        denoise_mask = torch.tensor(denoise_mask, device=device)
        denoise_indices = torch.where(denoise_mask == 1)[0]
        cond_indices = torch.where(denoise_mask == 0)[0]
        seq_length = denoise_mask.shape[0]

        latents = self.prepare_init_latents(
            batch_size * num_images_per_prompt,
            seq_length,
            num_channels_latents,
            height,
            width,
            self.dtype,
            device,
            generator,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # image_seq_len = latents.shape[1] * latents.shape[-1] * latents.shape[-2] / self.transformer.config.patch_size[-1] / self.transformer.config.patch_size[-2]
        image_seq_len = noise_scale * sum(denoise_mask) * latents.shape[-1] * latents.shape[-2] / self.transformer.config.patch_size[-1] / self.transformer.config.patch_size[-2]
        # image_seq_len = 256
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        if cond_latents is not None:
            latents[:, cond_indices] = cond_latents
        
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                input_t = torch.broadcast_to(einops.repeat(torch.Tensor([t]).to(device), "1 -> 1 f 1 1 1", f=latent_model_input.shape[1]), latent_model_input.shape).clone()
                
                input_t[:, cond_indices] = self.copy_scheduler.timesteps[-1]
                # input_t[:, cond_indices] = self.scheduler.timesteps[-1]

                # predict the noise residual
                noise_pred = self.transformer(
                    samples=latent_model_input.to(self.dtype),
                    timesteps=input_t,
                    encoder_hidden_states=encoder_hidden_states.to(self.dtype),
                    encoder_attention_mask=encoder_attention_mask,
                    **forward_kwargs
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                bs, n_frame = noise_pred.shape[:2]
                noise_pred = einops.rearrange(noise_pred, "b f c h w -> (b f) c h w")
                latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = einops.rearrange(latents, "(b f) c h w -> b f c h w", b=bs, f=n_frame)
                
                if cond_latents is not None:
                    latents[:, cond_indices] = cond_latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        decoded_latents = latents / 1.658
        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        if latents.ndim == 5:
            latents = latents[:, denoise_indices]
            latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")
        image = self.vae.decode(latents.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return ICRDragPipelineOutput(images=image, latents=decoded_latents)

    

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None):
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

        if image is None:
            # scale the initial noise by the standard deviation required by the scheduler
            # latents = latents * self.scheduler.init_noise_sigma
            return latents
        
        image = image.to(device=device, dtype=dtype)
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        elif isinstance(generator, list):
            if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                )
            init_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image.to(self.vae.dtype)), generator=generator)
            
        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents = init_latents.to(device=device, dtype=dtype)

        init_latents = einops.rearrange(init_latents, "(bs views) c h w -> bs views c h w", bs=batch_size, views=init_latents.shape[0]//batch_size)
        # latents = einops.rearrange(latents, "b c h w -> b 1 c h w")
        # latents = torch.concat([latents, init_latents], dim=1)
        return init_latents
    
    def prepare_init_latents(self, batch_size, seq_length, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, seq_length, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_path = pretrained_model_name_or_path
        cache_dir = kwargs.pop("cache_dir", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        for ignored_arg in (
            "force_download",
            "proxies",
            "local_files_only",
            "token",
            "revision",
            "from_flax",
            "torch_dtype",
            "custom_pipeline",
            "custom_revision",
            "provider",
            "sess_options",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
            "variant",
            "use_safetensors",
            "use_onnx",
            "load_connected_pipeline",
        ):
            kwargs.pop(ignored_arg, None)

        transformer = kwargs.pop("transformer", None)
        vae = kwargs.pop("vae", None)
        text_encoder = kwargs.pop("text_encoder", None)
        # transformer = kwargs.pop("transformer", None)
        
        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Using `device_map` requires the `accelerate` library. Please install it using: `pip install accelerate`."
            )

        if device_map is not None and not isinstance(device_map, str):
            raise ValueError("`device_map` must be a string.")

        if device_map is not None and device_map not in SUPPORTED_DEVICE_MAP:
            raise NotImplementedError(
                f"{device_map} not supported. Supported strategies are: {', '.join(SUPPORTED_DEVICE_MAP)}"
            )

        if device_map is not None and device_map in SUPPORTED_DEVICE_MAP:
            if is_accelerate_version("<", "0.28.0"):
                raise NotImplementedError("Device placement requires `accelerate` version `0.28.0` or later.")

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )
        if transformer is None:
            transformer = NextDiT.from_pretrained(f"{model_path}", subfolder="transformer", torch_dtype=torch.float32, cache_dir=cache_dir)
        if vae is None:
            vae = AutoencoderKL.from_pretrained(f"{model_path}", subfolder="vae", cache_dir=cache_dir)
        if text_encoder is None:
            text_encoder = T5EncoderModel.from_pretrained(f"{model_path}", subfolder="text_encoder", torch_dtype=torch.float16, cache_dir=cache_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer", cache_dir=cache_dir)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir=cache_dir)

        pipeline = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            **kwargs
        )

        return pipeline
