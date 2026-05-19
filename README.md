# Visual Product Recommendation

This repo contains

## Model Summary

## Data Processing & Cleaning

There is not much pre-processing required for this project as the transformers library handles a lot of this automatically. However the dataset does contain some empty images (size (1, 1)) so these have been removed from the dataset in production. There are also quite a lot of duplicate images, around 20,000 out of the 50,000 original images. To remove these I ran a python tool called find-dups and wrote a script to filter the duplicate and empty images in `src/data.py`.

## Improving Query Times

Originally I was simply loading the embeddings into a big list and iterating through calculating the cosine similarity between each vector and the query vector one by one. The first improvement I made to this is stacking the embeddings into one big tensor and computing the similarity vectorised. This speeds up queries dramatically, going from around 20 secs to 2 secs.

This also meant I could store the large tensor in one file instead of a bunch of individual files. This also speeds up loading time dramatically as it cuts down the I/O overhead to one file.

## Exploring ANN Methods

With the imrovements I made, query times were quite low already but I wanted to explore some more sophisticated methods. It also felt quite inefficient to calculate the cosine similarity on every query so I started researching how other recommendation systems work in production. I came across a familiy of techniques called Approximate Nearest Neighbors (ANN) and in particly Hierarchical Navigable Small Worlds (HNSW).

HNSW works by consructing an index structure, much like an index in a standard database that narrow down the search space for a given query vector. Vectors are added one by one and arranged in a graph structure by their relative inner product (equivilant to similarity for normalised vectors). The specifics of the method and implementation are beyond the scope of the project however I there are many libraries that implement the method, I'm using a popular one I found called nmslib.

I've kept the functionality for both methods, there's a selector in the frontend UI and the query time is displayed for easy comparison between methods. The HNSW method is considerably faster on all test images I've used. This comes at the cost of some accuracy and needing to define k.
