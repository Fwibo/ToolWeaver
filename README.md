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


### Datasets

We follow the data construction pipeline of [ToolGen](https://github.com/Reason-Wang/ToolGen). Our experiments are based on the **ToolBench** dataset.

The training data is processed into **ShareGPT-like format** and divided into three categories corresponding to the training stages. You can download the processed datasets from the [ToolGen HuggingFace Collection](https://huggingface.co/collections/reasonwang/toolgen-66f28e2079085526806509c2)


### Training

#### Stage 1: Structured Tokenization
```bash
cd index && bash run.sh                    # Basic training
python main_sim_loss.py --data_path ...    # With collaborative loss
python generate_indices_toolweaver.py      # Generate indices
```

### Stage 2: Generative Alignment
We adopt a multi-stage fine-tuning strategy, located in the `./train` folder.

1.  **Vocabulary Expansion**: Unlike ToolGen which adds atomic tokens (e.g., `<<ToolName>>`), we resize the tokenizer to include code tokens (e.g., `<a_12>`, `<b2_5>`) initialized from the VAE codebook.
2.  **Retrieval Training**: Train the model to generate the correct tool codes based on user queries.
3.  **End-to-End Agent-Tuning**: Fine-tune with full conversation trajectories to handle arguments and multi-turn interactions.

A sample data entry for ToolWeaver (Memorization Stage):
```json
{
    "conversations": [
        {
            "role": "user",
            "content": "Tool Name: QRCheck. Description: Check quality...",
            "loss": false
        },
        {
            "role": "assistant",
            "content": "<a_10><b_45><c_12><d_8>", 
            "loss": true
        }
    ]
}
```

### Evaluation

For detailed evaluation scripts and baselines, please refer to the [ToolGen Repository](https://github.com/Reason-Wang/ToolGen).

*   **Retrieval**: See `scripts/retrieval`.
*   **Pass Rate**: Use `scripts/pass_rate` to evaluate ToolBench test sets.
*   **Win Rate**: Use `scripts/preference` for comparisons.

### Citation

If our work or the ToolGen framework is helpful, please kindly cite:

```bibtex
[Your Citation Here]
```

