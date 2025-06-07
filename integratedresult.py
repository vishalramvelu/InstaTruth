import re
import random

def parse_deepseek_polarity(summary:str, confidence:float, verdict: str) -> float:
    """
    Convert DeepSeek's verdict and confidence into P(true).
    """

    webnumber = -1

    #If verdict is true, we can be confident at that level 
    if verdict == 'real':
        webnumber = confidence

    #If verdict is false, we can invert confidence of being false to P(true)
    elif verdict == 'fake':
        newer = 1 - confidence
        webnumber = newer
    
    #If verdict is debated/inconclusive, P(true) is around 0.5
    else:
        x = random.uniform(0.4,0.6)
        webnumber = x

    return webnumber

    
def combine_predictions(bert_out: dict, explanation: dict) -> dict:
    """
    Combine BERT output and DeepSeek explanation into a final P(false) and label.

    Args:
      bert_out (dict): {"label": "fake" or "real", "score": [0,1]}
      explanation (dict): {"summary": str, "confidence": [0,1], "verdict": str}

    Returns:
      {
        "combined_score": [0,1],   
        "final_label": "fake" or "real" or "inconclusive"
        "bert_score": [0,1]
        "fact-check score": [0,1]
      }
    """
    # 1) Extract BERT_score = P(false)
    blabel = bert_out.get("label", "").lower()
    bscore = bert_out.get("score", 0.0)
    if 0.25 < bscore < 0.75:
        bert_score = 0.5  #inconclusive confidence level
    elif blabel == "real":
        bert_score = bscore 
    else:
        if bscore > 0.90:
            bert_score = 1 - bscore  
        else:
            jokes = random.uniform(0.4,0.6) 
            bert_score = jokes

    # 2) Extract factcheck_score = P(true) using LLM summary
    summary    = explanation.get("summary", "")
    confidence = explanation.get("confidence", 0.0)
    verdict = explanation.get("verdict")
    factcheck_score = parse_deepseek_polarity(summary, confidence,verdict)

    # 3) Algo to combine BERT + Web search
    webweight, bertweight = 0.85, 0.15
    combined_score = -1
    if factcheck_score >= 0.80:
        combined_score = factcheck_score 
    elif factcheck_score <= 0.20:
        combined_score = factcheck_score 
    else:
        weighted = webweight*factcheck_score + bertweight*bert_score
        combined_score = weighted
        #combined_score = alpha * bert_score + (1.0 - alpha) * factcheck_score

    # 4) Find final label
    final_label = None

    if combined_score >= 0.67:
        final_label = "real"
    
    elif 0.33 < combined_score < 0.67:
        final_label = "inconclusive"

    else:
        final_label = "fake"

    return {
        "combined_score": round(combined_score, 4),
        "final_label": final_label,
        "bert_score": round(bert_score, 4),
        "factcheck_score": round(factcheck_score, 4)
    }