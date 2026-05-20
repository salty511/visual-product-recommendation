from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torch.nn.functional import normalize
import torch
import os
from transformers import AutoImageProcessor, AutoModel
from fastapi.staticfiles import StaticFiles
from time import perf_counter
import nmslib
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

DEVICE = os.getenv("device") or "cpu"

EMBEDDINGS_STACKED_PATH = os.path.join(os.path.dirname(__file__), f'../embeddings_stacked.pt')
EMBEDDING_NAMES_PATH = os.path.join(os.path.dirname(__file__), f'../embedding_names.txt')
INDEX_PATH = os.path.join(os.path.dirname(__file__), f'../hnsw_cosine_index.bin')
STATIC_PATH = os.path.join(os.path.dirname(__file__), './static')
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

if os.getenv("S3"):
    IMAGES_PREFIX = "https://visual-product-recommendations.s3.eu-north-1.amazonaws.com/clean-images/"
else:
    IMAGES_PATH = os.path.join(os.path.dirname(__file__), '../data/clean')
    app.mount("/images", StaticFiles(directory=IMAGES_PATH))
    IMAGES_PREFIX = "/images/"

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

torch.manual_seed(0)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)

model.eval()

index = nmslib.init(method='hnsw', space='cosinesimil')

with open(EMBEDDING_NAMES_PATH, "r") as f:
    embedding_names = f.readlines()

print(embedding_names[0])

def load_embeddings():
    print("Loading Embeds")
    E = torch.load(EMBEDDINGS_STACKED_PATH, map_location=torch.device(DEVICE))
    E = E.float()
    E = normalize(E, dim=1)
    index.loadIndex(INDEX_PATH)
    return E

def infer(image):
	inputs = processor(image, return_tensors="pt").to(model.device)
	with torch.no_grad():
		outputs = model(**inputs)
	emb = outputs.pooler_output.float()
	emb = normalize(emb, dim=1)
	return emb

def calculate_cosine_scores(em_1, E):
    print("Calculating Cosine Similarity")
    scores = (em_1 @ E.T).squeeze(0)
    mask = scores >= 0.7
    return {embedding_names[i][:-4]: scores[i].item() for i in mask.nonzero().flatten().tolist()}

E = load_embeddings()

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/search")
def search(file: UploadFile, mode: str = "index", k: int = 5):
    print(file.filename, file.content_type)
    im = Image.open(file.file).convert("RGB")
    em_1 = infer(im)
    
    res = {"scores": [], "time": 0}

    if(mode == "cosine"):
        start = perf_counter()

        cosine_scores = calculate_cosine_scores(em_1, E)
        cosine_scores = {k: v for k, v in sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)}

        for key in cosine_scores.keys():
            res["scores"].append({"src": f"{IMAGES_PREFIX}{key}", "score": cosine_scores[key]})

        end = perf_counter()
        res["time"] = end - start
        print(end - start) 
    else:
        start = perf_counter()

        ids, distances = index.knnQuery(em_1, k=k)
        
        for i in range(k):
            res["scores"].append({"src": f"{IMAGES_PREFIX}{embedding_names[ids[i]][:-4]}", "score": 1 - distances[i].item()})

        end = perf_counter()
        res["time"] = end - start
        print(end-start)

    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
