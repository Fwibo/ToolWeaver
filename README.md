# ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models

## Overview

**ToolWeaver** addresses two key limitations of current tool-augmented LLMs: scalability crisis (vocabulary size explosion) and semantic bottleneck (sparse tool relationship learning). Instead of mapping each tool to a unique token, ToolWeaver encodes tools into hierarchical sequences that enable logarithmic vocabulary growth and dense collaborative learning from shared code co-occurrence.

The framework consists of two stages: (1) **Structured Tokenization** that weaves tool semantics with co-usage patterns into hierarchical codes, and (2) **Generative Alignment** that fine-tunes LLMs to generate these codes. Evaluation on 47,000 tools shows significant improvements over state-of-the-art methods.

## Method

### Stage 1: Structured Tokenization
1. **Semantic Encoding**: Convert tool documentation to dense embeddings
2. **Collaborative-Aware RQ-VAE**: Multi-level quantization with graph Laplacian regularization to encourage similar tools to share codes
3. **Uniform Mapping**: Resolve collisions using Sinkhorn-Knopp optimal transport

### Stage 2: Generative Alignment  
1. **Retrieval Alignment**: Fine-tune LLM to generate hierarchical codes from queries
2. **Trajectory Alignment**: Train on complete interaction flows for end-to-end tool use


## Quick Start

ToolWeaver training follows a **two-stage pipeline**: first learning structured tool representations, then aligning them with LLMs for generative tool use.

### Prerequisites

The included `requirements.txt` provides basic dependencies. You may need to supplement it on your target machine:

```bash
# Install basic dependencies
pip install -r requirements.txt

```

See the "Requirements.txt Generation Guide" section below for more detailed dependency management.

### Data Preparation

1. **Tool embeddings**: `./data/ToolBench/toolweaver-mean-embeddings-*.npy`
2. **Similarity matrix**: `./data/similarity_matrix.pkl` 
3. **Query-tool pairs**: JSON files for retrieval alignment
4. **Interaction trajectories**: Complete conversation flows for trajectory alignment
5. **Base models**: Place pretrained LLMs in `./models/`

### Training

#### Structured Tokenization
```bash
cd index && bash run.sh                    # Basic training
python main_sim_loss.py --data_path ...    # With collaborative loss
python generate_indices_toolweaver.py      # Generate indices
```

#### Generative Alignment

The relevant code is located in the `./train` folder.


