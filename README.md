# Visual Product Recommendation

This repo contains

## Model Summary

## Data Processing & Cleaning

There is not much pre-processing required for this project as the transformers library handles a lot of this automatically. However the dataset does contain some empty images (size (1, 1)) so these have been removed from the dataset in production. There are also quite a lot of duplicate images, around 20,000 out of the 50,000 original images. To remove these I ran a python tool called find-dups and wrote a script to filter the duplicate and empty images in `src/data.py`.

## Improving Query Times

Originally I was simply loading the embeddings into a big list and iterating through calculating the cosine similarity between each vector and the query vector one by one. The first improvement I made to this is stacking the embeddings into one big tensor and computing the similarity vectorised. This speeds up queries dramatically, going from around 20 secs to 2 secs.

## Storing Embeddings

Originally I

## Exploring ANN Methods

In the initial stages of the project I was only working with a small subset of the images and simply calculating the cosine similarity on every query. It quickly became apparent that query times would explode as I included more of the original dataset. It also felt very inefficient to do it this way so I started researching how other recommendation systems work in production. I came across a familiy of techniques called Approximate Nearest Neighbors (ANN) and in particly Hierarchical Navigable Small Worlds (HNSW).

HNSW works by consructing an index structure, much like an index in a standard database that narrow down the search space for a given query vector. The specifics of the method and implementation are beyond the scope of the project however I there are many libraries that implement the method, I'm using a popular one I found called nmslib.
