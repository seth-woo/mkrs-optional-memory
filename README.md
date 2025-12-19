# mkrs-optional-memory
Multimodal Knowledge Retrieval System with Optional Memory (MKRS)
* MKRS is a multimodal AI system for image-based question answering with optional semantic memory.
* It supports both direct visual reasoning over single images and retrieval-augmented reasoning across multiple images using a self-hosted Qdrant vector database.

## Features
- Single-image visual question answering (stateless)
- Multi-image retrieval-augmented reasoning (planned)
- Open-source vision-language models
- Self-hosted Qdrant vector database

## To Run
```bash
pip install -r requirements.txt
```
```bash
uvicorn app.main:app --reload
```
After running the above commands, open your browser and navigate to `http://localhost:8000/docs`.
## License

This repository is licensed under the [MIT License](LICENSE.md).
