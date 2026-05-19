FROM python:3.11-slim

WORKDIR /app

# system deps for PIL and nmslib build
RUN apt-get update && apt-get install -y \
	build-essential \
	libjpeg62-turbo-dev \
	zlib1g-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api
COPY embeddings_stacked.pt /app/embeddings_stacked.pt
COPY embedding_names.txt /app/embedding_names.txt
COPY hnsw_cosine_index.bin /app/hnsw_cosine_index.bin

ENV device=cpu
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
