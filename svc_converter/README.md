# SVC Enhanced Dataset Converter for Grilly

Converts raw text datasets to structured SVC (Subject-Verb-Complement) format with full linguistic annotations for training Grilly's language models.

## Overview

This converter transforms your existing datasets into the rich SVC format needed for Grilly's experimental language features:

```
Raw text → spaCy transformer → SVC extraction → Full linguistic annotations
```

### Output Format

Each entry contains:
- **SVC structure**: Subject, Verb, Complement decomposition
- **Linguistic features**: Tokens, POS tags, lemmas, dependencies, named entities
- **Tagged versions**: Multiple annotation formats for different use cases
- **Structural features**: Syntactic complexity, verb tense, discourse markers
- **Semantic roles**: Agent, action, theme, domain classification

## Requirements

- **GPU**: A100/A10/RTX 4090 recommended (8GB+ VRAM)
- **RAM**: 32GB+ recommended for large files
- **Storage**: ~50GB for processing 4GB input → ~20GB output
- **Python**: 3.10+

## Quick Start (TensorDock/RunPod)

### 1. Spin up a GPU instance

Recommended specs:
- **GPU**: A100 40GB or A10 24GB
- **vCPUs**: 8+
- **RAM**: 32GB+
- **Storage**: 100GB SSD

### 2. Upload files and run setup

```bash
# Upload the converter scripts
# Then run setup
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### 3. Upload your data

Upload to `/workspace/data/input/`:
- `temporal_dataset.jsonl` (1.7GB)
- `instruct_anonymized_cleaned.json` (251MB)
- `conversations_dataset_anonymized_cleaned.jsonl` (62MB)

### 4. Run conversion

**Single GPU (simple):**
```bash
source /workspace/svc_env/bin/activate
cd /workspace/scripts

python convert_to_svc.py \
    --input /workspace/data/input \
    --output /workspace/data/output \
    --gpu 0 \
    --batch-size 100
```

**Multi-worker (faster, for multi-GPU):**
```bash
python convert_parallel.py \
    --input /workspace/data/input \
    --output /workspace/data/output \
    --workers 4 \
    --gpus 0,1,2,3
```

## Processing Time Estimates

Based on A100 40GB with batch_size=100:

| Dataset | Size | Est. Entries | Est. Time | Notes |
|---------|------|--------------|-----------|-------|
| `temporal_dataset.jsonl` | 1.7GB | ~1M | 8-12 hours | Historical articles |
| `temporal_dataset_events.jsonl` | 1.3GB | ~800K | 6-10 hours | Event data |
| `instruct_anonymized_cleaned.json` | 251MB | ~100K | 1-2 hours | Technical/code |
| `conversations_dataset.jsonl` | 62MB | ~50K | 30-60 min | Dialogues |

**Total estimate: 16-24 hours** on single A100

With 4x parallel workers: **~6-8 hours**

## Command Reference

### convert_to_svc.py (Single GPU)

```bash
python convert_to_svc.py \
    --input /path/to/data \      # Input directory or file
    --output /path/to/output \   # Output directory
    --gpu 0 \                    # GPU ID (-1 for CPU)
    --batch-size 100 \           # Batch size for spaCy
    --max-entries 1000 \         # Optional: limit entries (for testing)
    --model en_core_web_trf \    # spaCy model
    --files temporal_dataset.jsonl instruct.json  # Optional: specific files
```

### convert_parallel.py (Multi-GPU)

```bash
python convert_parallel.py \
    --input /path/to/data \
    --output /path/to/output \
    --workers 4 \                # Parallel workers
    --gpus 0,1,2,3 \             # GPU IDs to use
    --model en_core_web_trf
```

## Checkpointing & Resume

The converter automatically saves progress:
- Checkpoint files: `*.checkpoint` alongside output
- Resume: Just re-run the same command - already processed entries are skipped

## Output Structure

```
output/
├── temporal_dataset_svc_enhanced.jsonl
├── temporal_dataset_svc_enhanced.checkpoint
├── instruct_anonymized_cleaned_svc_enhanced.jsonl
├── instruct_anonymized_cleaned_svc_enhanced.checkpoint
└── conversations_dataset_anonymized_cleaned_svc_enhanced.jsonl
```

## Sample Output Entry

```json
{
  "id": "temporal_nyt_12345",
  "text": "The Roman Republic emerged after the overthrow of the monarchy.",
  "realm": "world/history",
  "language": "en",
  "metadata": {
    "svc": {
      "subject": "The Roman Republic",
      "verb": "emerged",
      "complement": "after the overthrow of the monarchy"
    },
    "domain": "World",
    "source": "nyt_archive",
    "date": "1852-01-01"
  },
  "linguistic_features": {
    "tokens": [...],
    "pos_tags": [...],
    "lemmas": [...],
    "dependencies": [...],
    "named_entities": [...]
  },
  "tagged_versions": {
    "svc_full_tagged": "[SUBJ]The Roman Republic[/SUBJ] [VERB]emerged[/VERB] [COMP]after the overthrow of the monarchy[/COMP]",
    "semantic_roles": {
      "agent": "The Roman Republic",
      "action": "emerged",
      "theme": "after the overthrow of the monarchy"
    }
  },
  "structural_features": {
    "verb_tense_info": [...],
    "syntactic_complexity": {...}
  }
}
```

## Cost Estimates (Cloud GPU)

| Provider | Instance | Cost/hr | Est. Total Cost |
|----------|----------|---------|-----------------|
| TensorDock | A100 40GB | ~$1.50 | ~$24-36 |
| RunPod | A100 40GB | ~$1.99 | ~$32-48 |
| Lambda | A100 40GB | ~$1.29 | ~$20-30 |

**Tip**: Use spot/interruptible instances with checkpointing for ~50% cost savings.

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 50 or 25
- Use a larger GPU or split processing

### Slow Processing
- Verify GPU is being used: check nvidia-smi during processing
- Increase batch size if GPU memory allows
- Use parallel processing with multiple workers

### spaCy Model Download Fails
```bash
pip install spacy[cuda12x]  # or cuda11x
python -m spacy download en_core_web_trf
```

### Resume After Crash
Just re-run the same command - checkpoint files track progress.

## Integration with Grilly

After conversion, copy outputs to your Grilly training directory:

```bash
cp /workspace/data/output/*_svc_enhanced.jsonl \
   /path/to/grilly/data/training/
```

Then update your Grilly experimental language config to point to the new data.

---

**GrillCheese AI** - Power without greed.
