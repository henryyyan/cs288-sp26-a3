#!/bin/bash
cat data_storage_100-20.json.part.* > data_storage_100-20.json
cat eecs_ind_100-20.faiss.part.* > eecs_ind_100-20.faiss
python3 src/RAG/generate.py "$1" "$2"
