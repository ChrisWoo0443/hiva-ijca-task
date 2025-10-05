# hiva-ijca

Fine-tuning DistilBERT on ELI5 dataset for Masked Language Modeling.

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- Python 3.9+

### Installation

1. Install dependencies using uv:
```bash
uv sync
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

## Training

Run the fine-tuning script:
```bash
python finetune.py
```

The script will:
- Download and process the ELI5 dataset
- Fine-tune DistilBERT for masked language modeling
- Save checkpoints to `./outputs/distilbert-finetuned-eli5-mlm`
- Log metrics to `./logs` for TensorBoard

## Monitoring

### TensorBoard

View training metrics in real-time:
```bash
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your browser.
