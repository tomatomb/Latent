from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
import numpy as np

# CLIP 모델 및 토크나이저 로드
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder", use_safetensors=True)

torch_device = "cuda"
text_encoder.to(torch_device)

prompt1 = "A watercolor painting of a Golden Retriever at the beach"
prompt2 = "A still life DSLR photo of a bowl of fruit"

# 두 개의 텍스트 프롬프트를 임베딩 벡터로 변환
text_input1 = tokenizer(prompt1, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings1 = text_encoder(text_input1.input_ids.to(torch_device))[0]

text_input2 = tokenizer(prompt2, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings2 = text_encoder(text_input2.input_ids.to(torch_device))[0]

# 임베딩 벡터를 보간
def interpolate_embeddings(embedding1, embedding2, steps):
    return [(1 - t) * embedding1 + t * embedding2 for t in np.linspace(0, 1, steps)]

interpolation_steps = 30
interpolated_embeddings = interpolate_embeddings(text_embeddings1, text_embeddings2, interpolation_steps)

# Stable Diffusion 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

# 보간된 벡터들을 이미지로 변환하여 저장
def embedding_to_image(embedding, step):
    with torch.no_grad():
        image = pipe(prompt=None, prompt_embeds=embedding, generator = generator).images[0]
        image.save(f"interpolated_image_{step}.png")

for step, embedding in enumerate(interpolated_embeddings):
    embedding_to_image(embedding, step)
