# InstaTruth

A hybrid fact-checking system that combines natural language processing and web search to verify the truthfulness of claims and statements made in short-form content (Tiktoks, Instagram Reels, Youtube Shorts, etc).

## Overview

InstaTruth uses a two-stage approach to fact-checking:
1. **NLP Analysis**: A BERT-based classifier provides initial fake/real classification
2. **Web Search Verification**: Google search results are analyzed by an LLM (DeepSeek) for fact-checking
3. **Combined Scoring**: Results from both stages are weighted and combined for a final verdict

## Features

### Core Analysis Pipeline
- **Video Processing**: Supports TikTok and Instagram video URLs with automatic download
- **Speech-to-Text**: OpenAI Whisper transcribes video audio with 99.1% accuracy
- **NLP Analysis**: Custom-trained 110M parameter DistilBERT model for semantic classification
- **Individual Claim Extraction**: Automatically identifies and analyzes specific factual claims
- **Multi-Step Web Verification**: Google Custom Search API integration with DeepSeek AI analysis
- **Source Credibility Weighting**: Evaluates supporting vs contradicting evidence from multiple sources

### Web Interface
- **Real-time Progress Tracking**: Live analysis progress with detailed step indicators
- **Access Key Authentication**: Secure beta access control system
- **Session Persistence**: Results persist across page reloads and browser sessions
- **Responsive Design**: Mobile-friendly interface with TailwindCSS
- **Comprehensive Results Display**: Individual claim analysis with evidence summaries

### Output & Scoring
- **Combined Confidence Scoring**: Weighted results from NLP and web verification
- **Three-Class Classification**: "Real", "Fake", or "Inconclusive" verdicts
- **Individual Claim Confidence**: Per-claim analysis with supporting/contradicting source counts
- **Evidence Summaries**: Detailed explanations for each factual determination

## Setup

1. Install dependencies:
- Requires `python3 --version == 3.11.3`
```bash
pip3 install -r requirements.txt
```

2. Create and setup `.env`:
```
DEEPSEEK_KEY=your-key
GOOG_CSE_ID=your-key
GOOG_KEY=your-key
ACCESS_KEY=access-key (you decide)
FLASK_SECRET=flask-secret (you decide)
```

3. Download model:
- Use `git lfs pull` to download complete `rfc_model.joblib`

## Usage

```bash
python3 app.py
```

## Troubleshooting
If you are having issues running the program, try:
- `brew install ffmpeg`
- `python3 fixcert.py`

and check
- `.env` setup
- `rfc_model.joblib` pulled using `git lfs`
