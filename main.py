from src.embeddings import infer
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from src.data import run_data_pipline
from src.embeddings import generate_embeddings
from PIL import Image
import matplotlib.pyplot as plt

load_dotenv()

DEVICE = os.getenv("device") or "cpu"

IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'data/clean')
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), 'embeddings_stacked.pt')
EMBEDDING_NAMES_PATH = os.path.join(os.path.dirname(__file__), 'embedding_names.txt')

def calculate_cosine_scores(em_1, E, embedding_names):
    print("Calculating Cosine Similarity")
    print(em_1.shape, E.shape)
    scores = cosine_similarity(em_1, E, dim=1)
    mask = scores > 0.5
    return {embedding_names[i][:-4]: scores[i].item() for i in mask.nonzero().flatten().tolist()}

def run_search_example():
    with open(EMBEDDING_NAMES_PATH, "r") as f:
        embedding_names = f.readlines()

    image_path = os.path.join("test_image.avif")
    E = torch.load(EMBEDDINGS_PATH, map_location=torch.device(DEVICE))

    em_1 = infer(image_path)

    cosine_scores = calculate_cosine_scores(em_1, E, embedding_names)

    cosine_scores = [[k, v] for k, v in sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)]
    
    k = 5 if len(cosine_scores) > 5 else len(cosine_scores)

    if k > 0:
        fig, axs = plt.subplots(1, k, figsize=(15, 9))

        im = np.asarray(Image.open(image_path))
        axs[0].imshow(im)
        axs[0].set_axis_off()
        
        for i in range(1, k):
            print(cosine_scores[i])
            im = np.asarray(Image.open(os.path.join(IMAGES_PATH, cosine_scores[i][0])))
            axs[i].imshow(im)
            axs[i].set_axis_off()
        
        plt.show()

if __name__ == "__main__":
    mode = os.getenv("mode") or "search"
    print(mode)

    if(mode == "setup"):
        run_data_pipline()
        generate_embeddings()
    run_search_example()
