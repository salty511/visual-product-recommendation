# Visual Product Recommendation

This repo contains

## Model Summary

## Data Processing & Cleaning

There is not much pre-processing required for this project as the transformers library handles a lot of this automatically. However the dataset does contain some empty images (size (1, 1)) so these have been removed from the dataset in production.

## Storing Embeddings

## Exploring ANN Methods

In the initial stages of the project I was only working with a small subset of the images and simply calculating the cosine similarity on every query. It quickly became apparent that query times would explode as I included more of the original dataset. It also felt very inefficient to do it this way so I started researching how other recommendation systems work in production. I came across a familiy of techniques called Approximate Nearest Neighbors (ANN) and in particly Hierarchical Navigable Small Worlds (HNSW).

HNSW works by consructing an index structure, much like an index in a standard database that narrow down the search space for a given query vector. The specifics of the method and implementation are beyond the scope of the project however I there are many libraries that implement the method, I'm using a popular one I found called nmslib.
