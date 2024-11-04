import torch
import os
import numpy as np
import time
from PIL import Image
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from transformers import logging

logging.set_verbosity_error()

print(torch.cuda.is_available())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

model_name_or_path = "runwayml/stable-diffusion-v1-5"

scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

pipe = StableDiffusionPipeline.from_pretrained(
    model_name_or_path,
    scheduler=scheduler,
    torch_dtype=torch.float32,
).to(device)

# Disable image generation progress bar, we'll display our own
pipe.set_progress_bar_config(disable=True)

# Offloading the weights to the CPU and only loading them on the GPU can reduce memory consumption to less than 3GB.
pipe.enable_model_cpu_offload()

# Tighter ordering of memory tensors.
pipe.unet.to(memory_format=torch.channels_last)

# Decoding large batches of images with limited VRAM or batches with 32 images or more by decoding the batches of latents one image at a time.
pipe.enable_vae_slicing()

# Splitting the image into overlapping tiles, decoding the tiles, and then blending the outputs together to compose the final image.
pipe.enable_vae_tiling()

# Using Flash Attention; If you have PyTorch >= 2.0 installed, you should not expect a speed-up for inference when enabling xformers.
pipe.enable_xformers_memory_efficient_attention()

seed = None

if seed is not None:
    generator = torch.manual_seed(seed)
else:
    generator = None

# The guidance scale is set to its normal range (7 - 10).
guidance_scale = 8

# The number of inference steps was chosen empirically to generate an acceptable picture within an acceptable time.
num_inference_steps = 15

# The higher you set this value, the smoother the interpolations will be. However, the generation time will increase. This value was chosen empirically.
num_interpolation_steps = 30

# I would not recommend less than 512 on either dimension. This is because this model was trained on 512x512 image resolution.
height = 512
width = 512

# The path where the generated GIFs will be saved
save_path = "output/output_circular"

if not os.path.exists(save_path):
    os.makedirs(save_path)

def display_images(images, save_path):
    try:
        # Convert each image in the 'images' list from an array to an Image object.
        images = [Image.fromarray(np.array(image[0], dtype=np.uint8)) for image in images]

        # Generate a file name based on the current time, replacing colons with hyphens
        # to ensure the filename is valid for file systems that don't allow colons.
        filename = time.strftime("%H:%M:%S", time.localtime()).replace(":", "-")
        # Save the first image in the list as a GIF file at the 'save_path' location.
        # The rest of the images in the list are added as subsequent frames to the GIF.
        # The GIF will play each frame for 100 milliseconds and will loop indefinitely.
        images[0].save(
            f"{save_path}/{filename}.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        for steps, image in enumerate(images):
            image.save(f"{save_path}/{filename}_step{steps}.png")
    except Exception as e:
        # If there is an error during the process, print the exception message.
        print(e)

    # Return the saved GIF as an IPython display object so it can be displayed in a notebook.
    #return IPdisplay.Image(f"{save_path}/{filename}.gif")

# The text prompt that describes the desired output image.
prompt = "Beautiful sea sunset, warm light, Aivazovsky style"
# A negative prompt that can be used to steer the generation away from certain features
negative_prompt = "picture frames"

# Generating initial latent vectors from a random normal distribution to create a loop interpolation between them.
latents = torch.randn(
    (2, 1, pipe.unet.config.in_channels, height // 8, width // 8),
    generator=generator,
)

# Calculation of looped embeddings
walk_noise_x = latents[0].to(device)
walk_noise_y = latents[1].to(device)

# Walking on a trigonometric circle
walk_scale_x = torch.cos(torch.linspace(0, 2, num_interpolation_steps) * np.pi).to(device)
walk_scale_y = torch.sin(torch.linspace(0, 2, num_interpolation_steps) * np.pi).to(device)

# Applying interpolation to noise
noise_x = torch.tensordot(walk_scale_x, walk_noise_x, dims=0)
noise_y = torch.tensordot(walk_scale_y, walk_noise_y, dims=0)

circular_latents = noise_x + noise_y

# Generating images using the interpolated embeddings.
images = []
for latent_vector in tqdm(circular_latents):
    images.append(
        pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latent_vector,
        ).images
    )

display_images(images, save_path)