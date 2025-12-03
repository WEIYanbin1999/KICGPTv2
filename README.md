# KICGPTv2: Large Language Model with Knowledge in Context for Knowledge Graph Completion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **KICGPTv2**, a framework that integrates Large Language Models with traditional Knowledge Graph Completion methods through in-context learning.

> **Paper**: KICGPTv2: Large Language Model with Knowledge in Context for Knowledge Graph Completion  
> **Authors**: Yanbin Wei, Qiushi Huang, James T. Kwok, Yu Zhang  
> **Published in**: IEEE Transactions on Knowledge and Data Engineering (TKDE)  
> **Previous Version**: [EMNLP 2023](https://github.com/WEIYanbin1999/KICGPT)

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Hyperparameters](#hyperparameters)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

KICGPTv2 addresses Knowledge Graph Completion (KGC) tasks through a **training-free** framework that combines:
- **Base KGC Models** (e.g., KG-BERT, LMKE) for preliminary ordering
- **Large Language Models** (Qwen-2.5-72B) for knowledge-enhanced re-ranking
- **In-Context Learning** with carefully designed demonstrations

### Supported Tasks

1. **Link Prediction**: Predict missing entities in `<h, r, ?>` or `<?, r, t>`
2. **Relation Prediction**: Predict missing relations in `<h, ?, t>`
3. **Triple Classification**: Determine if `<h, r, t>` is valid (Yes/No)

### Framework Stages

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: Preprocessing (Run Once)                               │
│  ├─ Entity Context-extraction → (description, aliases)          │
│  └─ Relation Self-alignment → refined relation descriptions     │
├─────────────────────────────────────────────────────────────────┤
│ Stage 1: Preliminary Ordering                                   │
│  └─ Base KGC model ranks all candidates → L_pre                 │
├─────────────────────────────────────────────────────────────────┤
│ Stage 2: Knowledge-prompting Re-ranking                         │
│  ├─ Load top-m candidates from L_pre                            │
│  ├─ Retrieve δ demonstrations (analogy + supplement)            │
│  ├─ Add entity contexts and relation alignments                 │
│  └─ LLM re-ranks → L_LLM                                         │
├─────────────────────────────────────────────────────────────────┤
│ Stage 3: Retrieval-augmented Reconstruction                     │
│  ├─ Map LLM outputs to candidate set (similarity > β)           │
│  └─ Combine with L_pre → L_KICGPTv2                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment Setup

### Requirements

- **Python**: 3.8 or higher
- **CUDA**: Optional (only for base KGC model training)

### Installation

```bash
# Clone the repository
git clone https://github.com/WEIYanbin1999/KICGPTv2.git
cd KICGPTv2

# Install dependencies
pip install openai tiktoken tqdm pyyaml

# (Optional) For base model training
pip install torch transformers
```

### Dependency Versions

```
openai>=0.27.0
tiktoken>=0.4.0
tqdm>=4.65.0
pyyaml>=6.0
```

### API Configuration

KICGPTv2 uses **Qwen-2.5-72B** via OpenAI-compatible API:

```bash
# Set API key (choose one method)
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Or pass via command line
--api_key YOUR_API_KEY
```

**API Endpoint**: Automatically configured to `https://api.pumpkinaigc.online/v1`

---

## Dataset Preparation

### Supported Datasets

- **fb15k-237**: Link Prediction, Relation Prediction
- **wn18rr**: Link Prediction, Relation Prediction  
- **fb13**: Triple Classification
- **umls**: Triple Classification, Link Prediction
- **wikidata5m**: Link Prediction (large-scale dataset)

### Directory Structure

```
datasets/
├── fb15k-237/
│   ├── entity2text.txt              # Entity names
│   ├── relation2text.txt            # Relation names
│   ├── get_neighbor/
│   │   ├── train2id.txt             # Training triples
│   │   ├── valid2id.txt             # Validation triples
│   │   ├── test2id.txt              # Test triples
│   │   ├── entity2id.txt            # Entity ID mapping
│   │   └── relation2id.txt          # Relation ID mapping
│   ├── entity_contexts.txt          # [GENERATED] Entity descriptions+aliases
│   ├── alignment/
│   │   └── alignment_clean.txt      # [GENERATED] Refined relations
│   └── demonstration/               # [GENERATED] Demonstration pools
├── wn18rr/
│   └── (similar structure)
├── fb13/
│   └── (similar structure)
├── umls/
│   └── (similar structure)
└── wikidata5m/
    └── (similar structure)
```

**Note**: Files marked with `[GENERATED]` are created during preprocessing (Steps 2-3 in Quick Start).
│   └── demonstration/               # [Generated] Demonstration pools
├── wn18rr/
│   └── (similar structure)
└── ...
```

### Download Datasets

All datasets must be downloaded separately:

1. **fb15k-237**: [https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
2. **wn18rr**: [https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
3. **fb13**: [https://github.com/wangpf3/LLM-for-KGC](https://github.com/wangpf3/LLM-for-KGC)
4. **umls**: [https://github.com/wangpf3/LLM-for-KGC](https://github.com/wangpf3/LLM-for-KGC)
5. **wikidata5m**: [https://deepgraphlearning.github.io/project/wikidata5m](https://deepgraphlearning.github.io/project/wikidata5m)

After downloading, place files in `datasets/{dataset_name}/` following the directory structure above.

---

## Quick Start

### Step 1: Generate Demonstrations

```bash
cd KICGPTv2
python get_demonstrations.py --dataset fb15k-237
```

**Output**: Creates demonstration pools in `datasets/fb15k-237/demonstration/`

### Step 2: Preprocessing

#### Entity Context-extraction

```bash
python entity_context_extraction.py \
    --dataset fb15k-237 \
    --api_key YOUR_API_KEY \
    --output_path ./datasets/fb15k-237/entity_contexts.txt
```

**Output**: `datasets/{dataset}/entity_contexts.txt` with format (one JSON per line):
```json
{"entity": "Q42", "entity_text": "Douglas Adams", "description": "English author", "aliases": ["Douglas Noel Adams"]}
{"entity": "Q5", "entity_text": "human", "description": "common name of Homo sapiens", "aliases": ["人類", "people"]}
```

#### Relation Self-alignment

```bash
# Step 1: Query LLM for relation alignment
python text_alignment_query.py \
    --dataset fb15k-237 \
    --api_key YOUR_API_KEY

# Step 2: Process and clean results
python text_alignment_process.py --dataset fb15k-237
```

**Output**: 
- `datasets/{dataset}/alignment/alignment_output.txt`: Raw LLM responses (one JSON per line)
- `datasets/{dataset}/alignment/alignment_clean.txt`: Cleaned relation dictionary (single JSON object)
  ```json
  {
    "/people/person/profession": "the occupation or job that [H] has",
    "/location/location/contains": "[H] geographically contains [T]"
  }
  ```

### Step 3: Task Execution

#### Link Prediction

```bash
python link_prediction.py \
    --dataset fb15k-237 \
    --query tail \
    --candidate_num 30 \
    --api_key YOUR_API_KEY \
    --eff_demon_step 16 \
    --similarity_beta 0.33 \
    --use_entity_context \
    --use_relation_alignment \
    --use_reconstruction
```

**Arguments**:
- `--query`: `tail` (predict `<h,r,?>`) or `head` (predict `<?,r,t>`)
- `--candidate_num`: Number of candidates to re-rank (m)
- `--eff_demon_step`: Number of demonstrations (δ)

#### Relation Prediction

```bash
python relation_prediction.py \
    --dataset fb15k-237 \
    --candidate_num 10 \
    --api_key YOUR_API_KEY \
    --eff_demon_step 8 \
    --use_entity_context \
    --use_relation_alignment \
    --use_reconstruction
```

#### Triple Classification

```bash
python triple_classification.py \
    --dataset fb13 \
    --api_key YOUR_API_KEY \
    --eff_demon_step 32 \
    --use_entity_context \
    --use_relation_alignment
```

### Step 4: Results

Output files are saved to `outputs/{dataset}/`:
- `output_{head|tail|relation|classification}.txt`: Final rankings
- `chat_{head|tail|relation|classification}.txt`: LLM conversation logs

---

## Hyperparameters

All hyperparameters are disclosed in our paper's Appendix and configured in `KICGPTv2/config.yaml`.

### Task-Specific Settings

| Task | Dataset | m (re-rank) | δ (demos) | β (threshold) |
|------|---------|-------------|-----------|---------------|
| **Link Prediction** | fb15k-237 | 30 | 16 | 0.33 |
| | wn18rr | 30 | 16 | 0.33 |
| **Relation Prediction** | fb15k-237 | 10 | 8 | 0.33 |
| | wn18rr | 10 | 8 | 0.33 |
| **Triple Classification** | fb13 | - | 32 | - |
| | umls | - | 32 | - |

### Key Parameters

- **m**: Number of top candidates to re-rank
- **δ**: Number of demonstrations (δ = eff_demon_step)
- **β**: Similarity threshold for reconstruction (default: 0.33)

### LLM Configuration

```yaml
model: "Qwen/Qwen2.5-72B-Instruct"
api_base: "https://api.pumpkinaigc.online/v1"
temperature: 0
max_tokens: 9600
max_llm_input_tokens: 3750
```

---

## Reproducibility

To ensure full reproducibility as promised to reviewers, we provide:

### 1. Complete Source Code
All Python scripts for:
- Preprocessing (`entity_context_extraction.py`, `text_alignment_*.py`)
- Task execution (`link_prediction.py`, `relation_prediction.py`, `triple_classification.py`)
- Supporting modules (`get_demonstrations.py`, `prompt_selection.py`)

### 2. Experiment Configurations

**Environment** (`KICGPTv2/config.yaml`):
- Python version: 3.8+
- Dependencies with versions
- LLM API settings

**Hyperparameters** (as disclosed in Appendix B):
- Link Prediction: m=30, δ=16, β=0.33
- Relation Prediction: m=10, δ=8, β=0.33
- Triple Classification: δ=32

### 3. Prompt Templates

All prompts are in `KICGPTv2/prompts/`:
- `entity_context.json`: Entity description extraction
- `link_prediction.json`: Link prediction prompts
- `relation_prediction.json`: Relation prediction prompts
- `triple_classification.json`: Triple classification prompts
- `text_alignment.json`: Relation self-alignment prompts

### 4. Step-by-Step Instructions

See [Quick Start](#quick-start) section above.

### 5. Dataset Links

- **fb15k-237 & wn18rr**: [Download](https://github.com/villmow/datasets_knowledge_embedding)
- **fb13 & umls**: Included in `datasets/`

---

## Advanced Usage

### Multi-Process Execution

For faster processing with multiple API keys:

```bash
# Create key file (one key per line)
cat > api_keys.txt << EOF
sk-key1xxxxx
sk-key2xxxxx
sk-key3xxxxx
EOF

# Run with parallel processes
python link_prediction.py \
    --dataset fb15k-237 \
    --query tail \
    --api_key api_keys.txt \
    --num_process 3
```

### Debug Mode

Test without API calls (manual input):
```bash
python link_prediction.py --dataset fb15k-237 --query tail --debug
```

Test with subset of data:
```bash
python link_prediction.py --dataset fb15k-237 --query tail \
    --api_key YOUR_KEY --debug_online
```

### Ablation Studies

**Disable entity contexts**:
```bash
python link_prediction.py --dataset fb15k-237 --query tail \
    --api_key YOUR_KEY --use_relation_alignment --use_reconstruction
```

**Disable relation alignment**:
```bash
python link_prediction.py --dataset fb15k-237 --query tail \
    --api_key YOUR_KEY --use_entity_context --use_reconstruction
```

**Disable reconstruction**:
```bash
python link_prediction.py --dataset fb15k-237 --query tail \
    --api_key YOUR_KEY --use_entity_context --use_relation_alignment
```

---

## Citation

If you use KICGPTv2 in your research, please cite:

```bibtex
@article{wei2024kicgptv2,
  title={KICGPTv2: Large Language Model with Knowledge in Context for Knowledge Graph Completion},
  author={Wei, Yanbin and Huang, Qiushi and Kwok, James T. and Zhang, Yu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024}
}
```

**Previous Conference Version**:
```bibtex
@inproceedings{wei2023kicgpt,
  title={KICGPT: Large Language Model with Knowledge in Context for Knowledge Graph Completion},
  author={Wei, Yanbin and Huang, Qiushi and Kwok, James T. and Zhang, Yu},
  booktitle={Proceedings of EMNLP 2023},
  year={2023}
}
```

---

## Contact

- **Yanbin Wei**: yanbin.ust@gmail.com
- **GitHub Issues**: [Create an issue](https://github.com/WEIYanbin1999/KICGPTv2/issues)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

This research was supported by the Hong Kong University of Science and Technology and Southern University of Science and Technology. We thank the reviewers for their valuable feedback that improved the reproducibility of our work.