#!/bin/bash
cat src/RAG/data_storage_100-20.json.part.* > src/RAG/data_storage_100-20.json
cat src/RAG/eecs_ind_100-20.faiss.part.* > src/RAG/eecs_ind_100-20.faiss
python3 src/RAG/generate.py "$1" "$2"
