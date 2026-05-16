from src.embeddings import infer
from tqdm.auto import tqdm
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
import os

def calculate_cosine_scores(em_1, embeddings):
    cosine_scores = {}
    for embedding in tqdm(embeddings):
        em_2 = torch.load(os.path.join("embeddings/clean", f"{embedding}"))
        similarity_score = cosine_similarity(em_1, em_2, dim=1).item()
        if(similarity_score > 0.7):
            cosine_scores[embedding[:-3]] = similarity_score
    return cosine_scores

if __name__ == "__main__":
    images = sorted(os.listdir("data/clean"))
    embeddings = sorted(os.listdir("embeddings/clean"))

    print(images[0])
    print(embeddings[0])

    embedding_files = sorted(os.listdir("embeddings/clean"))
    embedding_by_image = {f.replace(".pt", ""): f for f in embedding_files}

    image_name = sorted(os.listdir("data/clean"))[0]
    image_path = os.path.join("shopping.webp")

    em_1 = infer(image_path)

    cosine_scores = calculate_cosine_scores(em_1, embeddings)
    
    print(cosine_scores)

    cosine_scores = {k: v for k, v in sorted(cosine_scores.items(), key=lambda x: x[1], reverse=True)}

    print(cosine_scores)
