# Retrieval-Augmented Generation (RAG) with Hugging Face and Ollama

This project demonstrates how to build a simple Retrieval-Augmented Generation (RAG) system using Hugging Face datasets and models, FAISS for vector search, and Ollama for local large language model inference on macOS.

---

## Table of Contents

- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
  - [Ollama CLI](#ollama-cli)  
  - [Python Environment](#python-environment)  
- [Model Setup](#model-setup)  
- [Running the Notebook](#running-the-notebook)  
- [Usage](#usage)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)

---

## Prerequisites

- macOS machine (tested on macOS Ventura)  
- Python 3.10+ installed  
- Homebrew package manager (optional but recommended)  
- Internet connection for downloading models  

---

## Installation

### Ollama CLI

Ollama is required to run local LLMs and interact with them via the Python client.

1. **Install Ollama CLI:**

   Using Homebrew (recommended):


Or download the installer from the official website:  
[https://ollama.com/download/mac](https://ollama.com/download/mac)

2. **Verify installation:**

ollama --version

3. **Pull required models:**

ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest

4. **List installed models:**

ollama list

---

### Python Environment

1. **Create and activate a virtual environment (optional but recommended):**

python3 -m venv venv
source venv/bin/activate

2. **Install Python dependencies:**

pip install -r requirements.txt


---

## Model Setup

- The notebook uses the models pulled via Ollama CLI.  
- Ensure the model names used in the code match those listed by `ollama list`, including the `hf.co/` prefix and `:latest` suffix.

---

## Running the Notebook

1. **Start Jupyter Notebook or JupyterLab:**

jupyter notebook

2. **Open the `.ipynb` file in your browser.**

3. **Run cells sequentially:**  
- Load and chunk the dataset  
- Embed chunks and build FAISS index  
- Query and generate answers using Ollama models

---

## Usage

- Modify the `query` variable in the notebook to ask different questions.  
- Add or update documents in the knowledge base by editing the dataset or adding new chunks.  
- Experiment with prompt templates to improve answer quality.

---

## Troubleshooting

- **`ollama` command not found:**  
Make sure Ollama CLI is installed and your terminal PATH is updated. Restart your terminal if needed.

- **Model not found error:**  
Run `ollama pull <model-name>` to download the model locally.

- **Python import errors:**  
Ensure you installed all dependencies in the correct Python environment.

- **Naming conflicts:**  
Avoid naming your scripts `ollama.py` to prevent import issues.

---

## Contributing

Contributions and suggestions are welcome! Please open issues or pull requests on the repository.

---

## License

This project is licensed under the MIT License.

---

*Developed by Samie on macOS using RAG with Hugging Face and Ollama.*  
*For questions or help, contact: [your-email@example.com]*
