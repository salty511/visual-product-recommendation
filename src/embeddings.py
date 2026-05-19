from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv
import torch
from tqdm.auto import tqdm
import os
from PIL import Image
import nmslib
from torch.nn.functional import normalize

load_dotenv()

torch.manual_seed(0)

DEVICE = os.getenv("device") or "cpu"

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)

model.eval()

IMAGES_PATH = os.path.join(os.path.dirname(__file__), '../data/clean')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), f'../embeddings/{DEVICE}')

def _load_image(image_path):
    return Image.open(image_path).convert("RGB")

def infer(image_path):
    image = _load_image(image_path)
    inputs = processor(image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.pooler_output.float()
    return emb

def generate_embeddings():
    images = os.listdir("data/clean")
    images = sorted(images)

    for image in tqdm(images):
        embed = infer(os.path.join(IMAGES_PATH, image))
        torch.save(embed, os.path.join(EMBEDDINGS_PATH, f"{image}.pt"))

    embedding_names = sorted(os.listdir(EMBEDDINGS_PATH))

    with open(f"embedding_names.txt", "w") as f:
        f.writelines([f"{name}\n" for name in embedding_names])

    embed_map = {}
    
    for em in tqdm(embedding_names):
        embed_map[em] = torch.load(os.path.join(EMBEDDINGS_PATH, f"{em}"))

    E = torch.stack([embed_map[em].squeeze(0) for em in embedding_names]).float()

    torch.save(E, f"embeddings_stacked.pt")

    build_index(E.cpu())

def build_index(E):
    index = nmslib.init(method='hnsw', space='cosinesimil')
    E = normalize(E, dim=1)
    index.addDataPointBatch(E)
    index.createIndex()
    index.saveIndex('hnsw_cosine_index.bin')

if __name__ == "__main__":
    generate_embeddings()
