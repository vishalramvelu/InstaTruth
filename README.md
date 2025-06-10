# InstaTruth

A hybrid fact-checking system that combines natural language processing and web search to verify the truthfulness of claims and statements.

## Overview

InstaTruth uses a two-stage approach to fact-checking:
1. **NLP Analysis**: A BERT-based classifier provides initial fake/real classification
2. **Web Search Verification**: Google search results are analyzed by an LLM (DeepSeek) for fact-checking
3. **Combined Scoring**: Results from both stages are weighted and combined for a final verdict

## Features

- **BERT Classification**: Uses DistilBERT embeddings with Random Forest Classifier for initial text analysis
- **Web Search Integration**: Automatically searches Google for relevant sources
- **LLM Fact-Checking**: DeepSeek AI analyzes search results against the original claim
- **Confidence Scoring**: Provides probability scores and confidence levels for predictions
- **Three-Class Output**: Classifies claims as "real", "fake", or "inconclusive"

## Setup

1. Install dependencies:
- Requires `python3 --version == 3.11`
```bash
pip3 install -r requirements.txt
```

2. Create and setup `.env`:
```
DEEPSEEK_KEY=your-key
GOOG_CSE_ID=your-key
GOOG_KEY=your-key
```

3. Download model:
- Use `git lfs pull` for complete `rfc_model.joblib`

## Usage

### Basic Usage
```python

```

### Evaluate Transcript
```python
from instatruth import evaluate_transcript

evaluate_transcript("claim or statement to fact-check")
```

## API Dependencies

- **Google Custom Search API**: For retrieving relevant web sources
- **OpenRouter API**: For accessing DeepSeek AI model
- **Hugging Face Transformers**: For BERT model and text summarization