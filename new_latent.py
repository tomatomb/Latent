# 필요한 라이브러리 설치
#!pip install diffusers transformers accelerate torch torchvision --quiet

import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Stable Diffusion 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 텍스트 프롬프트
prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"

# 텍스트 인코딩 (latents로 변환)
with torch.no_grad():
    encoding_1 = pipe.tokenizer(prompt_1)
    encoding_2 = pipe.tokenizer(prompt_2)
    encoding_1 = pipe.text_encoder(torch.tensor(encoding_1))
    encoding_2 = pipe.text_encoder(torch.tensor(encoding_2))

# 인코딩 간 선형 보간(interpolation)
interpolation_steps = 5
interpolated_encodings = torch.linspace(0, 1, steps=interpolation_steps).to("cuda").unsqueeze(1) * encoding_1 + \
                         (1 - torch.linspace(0, 1, steps=interpolation_steps).to("cuda").unsqueeze(1)) * encoding_2

# 결과 이미지를 생성
images = []
for encoding in interpolated_encodings:
    image = pipe(prompt=None, latents=encoding).images[0]
    images.append(image)

# GIF 저장 함수
def export_as_gif(filename, images, frames_per_second=10, rubber_band=False):
    if rubber_band:
        images += images[2:-1][::-1]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )

# GIF로 변환하여 저장
export_as_gif(
    "doggo-and-fruit-5.gif",
    images,
    frames_per_second=2,
    rubber_band=True,
)

image.save(f"interpolated_image_{step}.png")
