import json
import os
import requests
import KEYS
from transformers import pipeline

# load keys & base endpoints
GOOGLE_CSE_ID = KEYS.GOOG_CSE_ID()
GOOGLE_API_KEY = KEYS.GOOG_KEY()
CSE_URL = "https://www.googleapis.com/customsearch/v1"

OPENROUTER_API_KEY = KEYS.DEEPSEEK_KEY()
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

# 1) Initialize a summarization pipeline (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_factual_claims(transcript: str) -> list:
    """
    Extract individual factual claims from transcript using enhanced prompting.
    
    Returns:
        List of claim dictionaries with claim text and context
    """
    if not OPENROUTER_API_KEY:
        # Fallback to simple summarization
        return [{"claim": summarize_transcript_simple(transcript), "context": "full_transcript"}]
    
    system_message = {
        "role": "system", 
        "content": (
            "You are a claim extraction specialist. Your job is to identify specific, "
            "verifiable factual claims from social media content.\n\n"
            "EXTRACT claims that are:\n"
            "• Specific facts, statistics, or assertions\n"
            "• Verifiable through credible sources\n"
            "• Not opinions, emotions, or subjective statements\n\n"
            "IGNORE:\n"
            "• Personal opinions (\"I think\", \"I believe\")\n"
            "• Emotional expressions\n"
            "• Vague statements without specific facts\n\n"
            "Return EXACTLY one JSON array of objects, nothing else:\n"
            "[\n"
            "  {\"claim\": \"specific factual statement\", \"type\": \"statistic|event|assertion\"},\n"
            "  ...\n"
            "]\n\n"
            "If no verifiable claims exist, return: []"
        )
    }
    
    user_message = {
        "role": "user",
        "content": f"Extract factual claims from this transcript:\n\n\"{transcript}\""
    }
    
    try:
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
            raise Exception(f"DeepSeek error {r.status_code}")
        
        content = r.json()["choices"][0]["message"]["content"].strip()
        
        # Clean up response
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        # Parse JSON
        claims = json.loads(content)
        return claims if isinstance(claims, list) else []
        
    except Exception as e:
        print(f"Claim extraction failed: {e}")
        # Fallback to simple summarization
        return [{"claim": summarize_transcript_simple(transcript), "type": "general"}]

def summarize_transcript_simple(transcript: str, max_length: int = 80) -> str:
    """
    Simple transcript summarization using BART.
    """
    try:
        summary_list = summarizer(
            transcript,
            max_length=max_length,
            min_length=int(max_length * 0.5),
            do_sample=False
        )
        return summary_list[0]["summary_text"]
    except Exception as e:
        print(f"Summarization failed: {e}")
        # Return first 100 words as fallback
        words = transcript.split()[:100]
        return " ".join(words)

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
    
    newquery = summarize_transcript_simple(transcript, max_length=80)

    
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

def fact_check_single_claim(claim: str, results: list) -> dict:
    """
    Fact-check a single claim against search results with enhanced prompting.
    
    Args:
        claim: The specific factual claim to verify
        results: List of search result dicts
    
    Returns:
        {
            "claim": str,
            "verdict": "real"|"fake"|"inconclusive", 
            "confidence": float,
            "evidence": str,
            "sources_supporting": int,
            "sources_contradicting": int
        }
    """
    if not OPENROUTER_API_KEY:
        return {
            "claim": claim,
            "verdict": "inconclusive",
            "confidence": 0.5,
            "evidence": "API unavailable",
            "sources_supporting": 0,
            "sources_contradicting": 0
        }
    
    # Format results with source credibility indicators
    formatted_results = ""
    for idx, item in enumerate(results, start=1):
        domain = item.get('displayLink', '')
        # Simple credibility scoring based on domain
        credibility = "HIGH" if any(x in domain for x in ['.gov', '.edu', 'reuters', 'ap.org', 'bbc']) else "MEDIUM"
        
        formatted_results += (
            f"{idx}. [{credibility} CREDIBILITY] {item['title']}\n"
            f"   Source: {domain}\n"
            f"   Content: {item['snippet']}\n"
            f"   URL: {item['link']}\n\n"
        )
    
    system_message = {
        "role": "system",
        "content": (
            "You are an expert fact-checker. Analyze the claim against provided sources and determine:\n\n"
            "VERDICT CRITERIA:\n"
            "• \"real\" - Multiple credible sources directly support the claim\n"
            "• \"fake\" - Multiple credible sources directly contradict the claim\n"
            "• \"inconclusive\" - Insufficient evidence, conflicting sources, or claim cannot be verified\n\n"
            "CONFIDENCE SCORING:\n"
            "• 0.9-1.0: Multiple high-credibility sources all agree\n"
            "• 0.7-0.8: Good evidence from credible sources\n"
            "• 0.5-0.6: Some evidence but limited or conflicting\n"
            "• 0.2-0.4: Weak evidence or mostly contradictory\n"
            "• 0.0-0.1: Strong evidence against the claim\n\n"
            "IMPORTANT RULES:\n"
            "• If NO sources directly address the claim → verdict=\"inconclusive\", confidence ≤ 0.4\n"
            "• Weight HIGH credibility sources more heavily\n"
            "• Look for DIRECT evidence, not tangential mentions\n"
            "• Be conservative - prefer \"inconclusive\" when uncertain\n\n"
            "Return EXACTLY one JSON object:\n"
            "{\n"
            "  \"verdict\": \"real\"|\"fake\"|\"inconclusive\",\n"
            "  \"confidence\": 0.0-1.0,\n"
            "  \"evidence\": \"2-3 sentence summary of evidence found\",\n"
            "  \"sources_supporting\": number_of_sources_supporting_claim,\n"
            "  \"sources_contradicting\": number_of_sources_contradicting_claim\n"
            "}"
        )
    }
    
    user_message = {
        "role": "user",
        "content": (
            f"CLAIM TO VERIFY: \"{claim}\"\n\n"
            f"SEARCH RESULTS ({len(results)} sources):\n{formatted_results}\n"
            "Analyze the claim against these sources and return your assessment."
        )
    }
    
    try:
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
            raise Exception(f"DeepSeek error {r.status_code}")
        
        content = r.json()["choices"][0]["message"]["content"].strip()
        
        # Clean response
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        # Parse and validate response
        parsed = json.loads(content)
        
        # Validate required fields
        result = {
            "claim": claim,
            "verdict": parsed.get("verdict", "inconclusive").lower(),
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            "evidence": parsed.get("evidence", "No analysis provided"),
            "sources_supporting": max(0, int(parsed.get("sources_supporting", 0))),
            "sources_contradicting": max(0, int(parsed.get("sources_contradicting", 0)))
        }
        
        # Validate verdict
        if result["verdict"] not in ["real", "fake", "inconclusive"]:
            result["verdict"] = "inconclusive"
            result["confidence"] = min(result["confidence"], 0.4)
        
        return result
        
    except Exception as e:
        print(f"Fact-checking failed for claim '{claim[:50]}...': {e}")
        return {
            "claim": claim,
            "verdict": "inconclusive", 
            "confidence": 0.3,
            "evidence": f"Analysis failed: {str(e)[:100]}",
            "sources_supporting": 0,
            "sources_contradicting": 0
        }

def multi_step_fact_check(transcript: str) -> dict:
    """
    Enhanced multi-step fact-checking pipeline.
    
    1. Extract individual factual claims
    2. Search for evidence for each claim
    3. Fact-check each claim separately  
    4. Aggregate results into overall assessment
    """
    try:
        # Step 1: Extract claims
        claims = extract_factual_claims(transcript)
        
        if not claims:
            return {
                "summary": "No verifiable factual claims detected in the transcript.",
                "verdict": "inconclusive",
                "confidence": 0.4,
                "claims_analyzed": 0,
                "individual_results": []
            }
        
        # Step 2 & 3: Fact-check each claim
        individual_results = []
        for claim_obj in claims:
            claim_text = claim_obj.get("claim", "")
            if not claim_text:
                continue
                
            # Search for this specific claim
            search_results = google_search(claim_text, num_results=5)
            
            # Fact-check the claim
            fact_check_result = fact_check_single_claim(claim_text, search_results)
            individual_results.append(fact_check_result)
        
        if not individual_results:
            return {
                "summary": "Could not analyze any claims from the transcript.",
                "verdict": "inconclusive", 
                "confidence": 0.3,
                "claims_analyzed": 0,
                "individual_results": []
            }
        
        # Step 4: Aggregate results
        total_claims = len(individual_results)
        real_claims = sum(1 for r in individual_results if r["verdict"] == "real")
        fake_claims = sum(1 for r in individual_results if r["verdict"] == "fake")
        inconclusive_claims = sum(1 for r in individual_results if r["verdict"] == "inconclusive")
        
        # Calculate weighted scores
        confidence_scores = [r["confidence"] for r in individual_results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Determine overall verdict
        if fake_claims > real_claims and fake_claims >= total_claims * 0.4:
            overall_verdict = "fake"
            overall_confidence = min(0.8, avg_confidence)
        elif real_claims > fake_claims and real_claims >= total_claims * 0.4:
            overall_verdict = "real"
            overall_confidence = min(0.8, avg_confidence)
        else:
            overall_verdict = "inconclusive"
            overall_confidence = max(0.3, min(0.6, avg_confidence))
        
        # Generate summary
        summary_parts = []
        if real_claims > 0:
            summary_parts.append(f"{real_claims} claims supported by evidence")
        if fake_claims > 0:
            summary_parts.append(f"{fake_claims} claims contradicted by evidence")
        if inconclusive_claims > 0:
            summary_parts.append(f"{inconclusive_claims} claims could not be verified")
        
        summary = f"Analyzed {total_claims} factual claims: {', '.join(summary_parts)}."
        
        return {
            "summary": summary,
            "verdict": overall_verdict,
            "confidence": round(overall_confidence, 4),
            "claims_analyzed": total_claims,
            "claims_real": real_claims,
            "claims_fake": fake_claims,
            "claims_inconclusive": inconclusive_claims,
            "individual_results": individual_results
        }
        
    except Exception as e:
        print(f"Multi-step fact-checking failed: {e}")
        return {
            "summary": f"Fact-checking pipeline failed: {str(e)[:100]}",
            "verdict": "inconclusive",
            "confidence": 0.2,
            "claims_analyzed": 0,
            "individual_results": []
        }
