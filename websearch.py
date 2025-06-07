import json
import os
import requests
from transformers import pipeline

# 1) Initialize a summarization pipeline (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_transcript(transcript: str, max_length: int = 100) -> str:
    """
    Summarize tiktok/video down to 'max_length' tokens (words).
    """
    summary_list = summarizer(
        transcript,
        max_length=80,
        min_length=int(max_length * 0.5),  # at least half as long
        do_sample=False
    )
    return summary_list[0]["summary_text"]



# env variable to access search engine
GOOGLE_CSE_ID  = '4202d4528cf704c22'
GOOGLE_API_KEY = 'AIzaSyB-YYb3YOrAXKs_ilEfG1jNngHhx_SUQ68'
CSE_URL        = "https://www.googleapis.com/customsearch/v1"

def google_search(query: str, num_results: int = 5):
    """
    Query Google Custom Search and return a list of up to 'num_results' dicts:
      {
        "title":       <page title>,
        "snippet":     <brief snippet>,
        "link":        <actual URL>,
        "displayLink": <short domain>,
        "formattedUrl":<user‐friendly URL>
      }
    """
    if not GOOGLE_CSE_ID or not GOOGLE_API_KEY:
        raise RuntimeError("Missing GOOGLE_CSE_ID or GOOGLE_API_KEY in env.")

    params = {
        "key": GOOGLE_API_KEY,
        "cx":  GOOGLE_CSE_ID,
        "q":   query,
        "num": num_results
    }

    resp = requests.get(CSE_URL, params=params)

    if resp.status_code != 200:
        raise RuntimeError(f"Google CSE error {resp.status_code}: {resp.text}")

    data = resp.json()
    items = data.get("items", [])
    results = []
    for item in items:
        results.append({
            "title":        item.get("title"),
            "snippet":      item.get("snippet"),
            "link":         item.get("link"),
            "displayLink":  item.get("displayLink"),
            "formattedUrl": item.get("formattedUrl")
        })
    return results



# Env varaible for LLM (deepseek ai)
OPENROUTER_API_KEY = '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'
OPENROUTER_URL     = 'https://openrouter.ai/api/v1/chat/completions'

def deepseek_reason_over_results(results: list, claim_text: str, num_sources: int = 5):
    """
    Given:
      - results: a list of dicts {title, snippet, link}
      - claim_text: the claim to fact‐check

    Build a prompt and call DeepSeek to return a summary + claim likliness

    Returns a dict in this format:
      {
        "summary": "…",
        "confidence": 0.85
        "verdict": 'real'
      }
    """

    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY in env.")

    # 1) Build the messages/prompt
    #    We want to give DeepSeek a structured list of results + the claim text.
    #    Then instruct it to “summarize” in 2-3 sentences and give a confidence score.
    formatted_results = ""
    for idx, item in enumerate(results, start=1):
        formatted_results += (
            f"{idx}. Title: {item['title']}\n"
            f"   Snippet: {item['snippet']}\n"
            f"   URL: {item['link']}\n\n"
        )

    system_message = {
        "role": "system",
        "content": (
            "You are a fact‐checking assistant. "
            "Given a claim and a set of news‐site snippets, decide if the claim is:\n"
            "  • TRUE (generally all real) → output verdict=\"real\"\n"
            "  • FALSE (no evidence supporting) → output verdict=\"fake\"\n"
            "  • or INCONCLUSIVE/DEBATED (contrasting viewpoints) → output verdict=\"debated\"\n\n"
            "Then output EXACTLY one JSON object, nothing else, with these keys:\n"
            "  \"summary\": a 2–3 sentence explanation,\n"
            "  \"confidence\": a float between 0.0 and 1.0 (your confidence in the final verdict provided),\n"
            "  \"verdict\": one of \"real\", \"fake\", or \"debated\"."
        )
    }


    user_message = {
        "role": "user",
        "content": (
            f"Claim to verify:\n\"{claim_text}\"\n\n"
            f"Below are {num_sources} news results:\n\n{formatted_results}\n"
            "Based on these, return that JSON."
        )
    }

    payload = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [system_message, user_message]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(OPENROUTER_URL, json=payload, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"DeepSeek error {r.status_code}: {r.text}")

    content = r.json()["choices"][0]["message"]["content"].strip()

    # strip out triple-backtick 
    if content.startswith("```"):
        # remove the first and last lines if they are backticks
        lines = content.splitlines()
        # detect if the first line is ``` or ```json
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # 2) Parse the JSON that DeepSeek returned
    try:
        parsed = json.loads(content)
        return {
            "summary": parsed.get("summary"),
            "confidence": parsed.get("confidence"),
            "verdict": parsed.get("verdict")
        }
    except json.JSONDecodeError:
        # In case it didn’t return pure JSON
        raise ValueError(f"Expected JSON but DeepSeek replied:\n{content}")


#way to extract results and display/process  
def get_factcheck_results(transcript: str, num_results: int = 5):
    """
    1) Summarize the transcript down to a short query.
    2) Call google_search(...) to get top N articles.
    3) Call deepseek_reason_over_results(...) on those articles + full transcript.
    4) Return a dict with "articles" and "explanation".
    """
    
    newquery = summarize_transcript(transcript, max_length=80)

    
    # 3) Fetch top articles
    articles = google_search(newquery, num_results=num_results)
    
    # 4) Run DeepSeek reasoning over those results + full transcript
    explanation = deepseek_reason_over_results(
        results=articles,
        claim_text=transcript,
        num_sources=len(articles)
    )
    
    # 5) Return both pieces in json format 
    return {
        "articles":    articles,
        "explanation": explanation
    }
