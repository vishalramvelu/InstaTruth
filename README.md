# InstaTruth

A hybrid fact-checking system that combines natural language processing and web search to verify the truthfulness of claims and statements made in short-form content (Tiktoks, Instagram Reels, Youtube Shorts, etc).

## Visit the Site
Feel free to check out the [website here!](https://instatruth.app/)

<img width="1469" alt="Screenshot 2025-06-25 at 10 59 54 AM" src="https://github.com/user-attachments/assets/86789e37-2672-438f-86e4-c204e9251dd2" />

## Framework
* Backend: Python, Flask, Scikit-learn, PyTorch, OpenAI Whisper, DeepSeek API, Google Custom Search API
* Frontend: Javascript, React, Tailwind CSS, HTML
* Deployment: Docker, Railway 

## Overview

InstaTruth uses a two-stage approach to fact-checking:
1. **NLP Analysis**: A BERT-based classifier provides initial fake/real classification
2. **Web Search Verification**: Google search results are analyzed by an LLM (DeepSeek) for fact-checking
3. **Combined Scoring**: Results from both stages are weighted and combined for a final verdict

## Features

* NLP Analysis using 110M paramter DistilBERT model for semantic classifcation
* Speech-to-Text using OpenAI Whisper transcribes video audio with 99.1 accuracy
* Multi-Step Web Verification using Google custom search API with DeepSeek analysis
* Detailed evidence summaries for each claim citing reasons for final output with sources linked

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

## Contributing
Contributions are welcome! If you'd like to enhance this website or report any issues, please submit a pull request or open an issue. 
