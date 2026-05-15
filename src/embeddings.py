from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv
import torch
from tqdm.auto import tqdm
import os
from PIL import Image
import random
import numpy as np

torch.manual_seed(0)

load_dotenv()

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224", device_map="auto")

model.eval()

def _load_image(image_path):
	return Image.open(image_path).convert("RGB")

def infer(image_path):
	image = _load_image(image_path)
	inputs = processor(image, return_tensors="pt").to(model.device)
	with torch.no_grad():
		outputs = model(**inputs)
	return outputs.pooler_output

if __name__ == "__main__":
	images = os.listdir("data/clean")
	images = sorted(images)

	for image in tqdm(images):
		embed = infer(os.path.join("data/clean", image))
		torch.save(embed, os.path.join("embeddings/clean", f"{image}.pt"))
