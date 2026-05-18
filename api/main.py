from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tqdm.auto import tqdm
from torch.nn.functional import cosine_similarity, normalize
import torch
import os
from transformers import AutoImageProcessor, AutoModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

IMAGES_PATH = os.path.join(os.path.dirname(__file__), '../data/clean')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), '../embeddings/clean')
STATIC_PATH = os.path.join(os.path.dirname(__file__), './static')
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app.mount("/images", StaticFiles(directory=IMAGES_PATH))
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

torch.manual_seed(0)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224", device_map="auto")

model.eval()

embedding_names = os.listdir(EMBEDDINGS_PATH)

def load_embeddings():
    embed_map = {}
    print("Loading Embeds")
    for em in tqdm(embedding_names):
        embed_map[em] = torch.load(os.path.join(EMBEDDINGS_PATH, f"{em}"))
    E = torch.stack([embed_map[em].squeeze(0) for em in embedding_names])
    return E

def infer(image):
	inputs = processor(image, return_tensors="pt").to(model.device)
	with torch.no_grad():
		outputs = model(**inputs)
	return outputs.pooler_output

def calculate_cosine_scores(em_1, E):
    print("Calculating Cosine Similarity")
    print(em_1.shape, E.shape)
    scores = cosine_similarity(em_1, E, dim=1)
    mask = scores > 0.7
    return {embedding_names[i][:-3]: scores[i].item() for i in mask.nonzero().flatten().tolist()}

E = load_embeddings()

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/search")
def search(file: UploadFile):
    print(file.filename, file.content_type)
    im = Image.open(file.file).convert("RGB")
    em_1 = infer(im)

    cosine_scores = calculate_cosine_scores(em_1, E)
    cosine_scores = {k: v for k, v in sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)}

    res = {"scores": []}

    for key in cosine_scores.keys():
        res["scores"].append({"src": key, "score": cosine_scores[key]})

    print(cosine_scores)

    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
