# RAG Pipeline for Install base

This repository provides a Retrieval-Augmented Generation (RAG) pipeline for processing and utilizing install bases for servers.

## Prerequisites

Before using this project, ensure you have the following dependencies installed:

- **[ChromaDB](https://github.com/chroma-core/chroma):** A vector database for storing embeddings.
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python):** Python bindings for running LLaMA-based models locally.

For ppc64le you can use these commands to get chroma, llama.cpp.python and other libraries:
micromamba create -n env python=3.10

micromamba install -c rocketce -c defaults pytorch-cpu scikit-learn pyyaml httptools onnxruntime "pandas<1.6.0" tokenizers

pip install -U --extra-index-url https://repo.fury.io/mgiessing --prefer-binary chromadb transformers psutil langchain sentence_transformers gradio==3.50.2 llama-cpp-python docling




## Installation

Install the other libraries with pip for x86 and with conda (rocketce or defaults as the channel)

## Usage

### 1. Convert the pdf files into a markdown file

1. Run the `install_base_processor.py` script:
   ```bash
   python install_base_processor.processor.py
   ```

### 2. Configure the LLM

To use the Large Language Model with the context from the vector DB (LLM):

1. Open `run_model.py` in your preferred text editor.
2. Update the `model:path` variable to point to your **GGUF model**.

### 3. Run the LLM

Execute the pipeline by running:
```bash
python run_model.py
```

## Folder Structure

- `/db`: Contains the vector database with collections.
- `install_base_processor.py`: Script for creating the vector database.
- `run_model.py`: Script for running the RAG pipeline using the configured LLM.

## Notes
- Login to server with this command: ssh -L 8082:localhost:8082 user@host_IP to allow for interaction with Chatbot/Gradio UI
- Make sure the GGUF model is compatible with `llama-cpp-python`.

## Contributing

If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. 

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this project.

---

Happy experimenting with the RAG Pipeline for RedBooks!

