import re
import random

def parse_deepseek_polarity(summary: str, confidence: float, verdict: str) -> float:
    """
    Convert DeepSeek's verdict and confidence into P(true).
    
    Args:
        summary: DeepSeek's explanation
        confidence: DeepSeek's confidence in its verdict (0-1)
        verdict: "real", "fake", "debated", or "inconclusive"
    
    Returns:
        P(true) as float between 0 and 1
    """
    # Validate inputs
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = 0.5  # Default to uncertain
    
    verdict = verdict.lower() if verdict else "inconclusive"
    
    if verdict in ['real', 'true']:
        # If verdict is real with confidence X, P(true) = X
        return confidence
    elif verdict in ['fake', 'false']:
        # If verdict is fake with confidence X, P(true) = 1-X
        return 1 - confidence
    else:  # debated, inconclusive, or unknown
        # For inconclusive cases, P(true) should be around 0.5
        # Use confidence to determine how far from 0.5 we deviate
        deviation = (confidence - 0.5) * 0.2  # Max deviation of 0.1
        return max(0.4, min(0.6, 0.5 + deviation))

def validate_prediction_scores(bert_out: dict, explanation: dict) -> tuple:
    """
    Validate and normalize input scores to ensure consistency.
    
    Returns:
        (bert_p_true, factcheck_p_true, bert_confidence, factcheck_confidence)
    """
    # Extract BERT scores
    bert_label = bert_out.get("label", "").lower()
    bert_score = bert_out.get("score", 0.5)  # P(true) from BERT
    bert_confidence = bert_out.get("confidence", bert_score)
    
    # Ensure BERT score is P(true)
    if bert_label == "fake" and bert_score > 0.5:
        # If labeled fake but score > 0.5, it might be P(fake)
        bert_score = 1 - bert_score
    
    # Extract factcheck scores
    factcheck_confidence = explanation.get("confidence", 0.5)
    factcheck_verdict = explanation.get("verdict", "inconclusive")
    factcheck_score = parse_deepseek_polarity(
        explanation.get("summary", ""), 
        factcheck_confidence, 
        factcheck_verdict
    )
    
    return (
        max(0, min(1, bert_score)),
        max(0, min(1, factcheck_score)),
        max(0, min(1, bert_confidence)),
        max(0, min(1, factcheck_confidence))
    )

    
def combine_predictions(bert_out: dict, explanation: dict) -> dict:
    """
    Combine BERT output and DeepSeek explanation into a final prediction.
    
    All scores now consistently represent P(true) for interpretability.

    Args:
        bert_out (dict): {"label": str, "score": float, "confidence": float}
        explanation (dict): {"summary": str, "confidence": float, "verdict": str}

    Returns:
        {
            "combined_score": float,     # Final P(true) 0-1
            "final_label": str,          # "real", "fake", or "inconclusive"
            "bert_score": float,         # BERT P(true) 0-1
            "factcheck_score": float,    # Web search P(true) 0-1
            "confidence": float          # Overall confidence 0-1
        }
    """
    # 1) Validate and normalize input scores
    bert_p_true, factcheck_p_true, bert_conf, factcheck_conf = validate_prediction_scores(
        bert_out, explanation
    )
    
    # 2) Dynamic weighting based on component confidence
    # Higher confidence components get more weight
    total_conf = bert_conf + factcheck_conf
    if total_conf > 0:
        bert_weight = bert_conf / total_conf * 0.3  # BERT max weight: 30%
        web_weight = factcheck_conf / total_conf * 0.7  # Web max weight: 70%
    else:
        bert_weight, web_weight = 0.15, 0.85  # Default weights
    
    # Normalize weights
    total_weight = bert_weight + web_weight
    if total_weight > 0:
        bert_weight /= total_weight
        web_weight /= total_weight
    
    # 3) Handle extreme confidence cases
    if factcheck_conf >= 0.85 and (factcheck_p_true <= 0.15 or factcheck_p_true >= 0.85):
        # High confidence web result dominates
        combined_score = factcheck_p_true
        overall_confidence = factcheck_conf
    elif bert_conf >= 0.85 and factcheck_conf <= 0.5:
        # High confidence BERT when web is uncertain
        combined_score = 0.7 * factcheck_p_true + 0.3 * bert_p_true
        overall_confidence = (bert_conf + factcheck_conf) / 2
    else:
        # Standard weighted combination
        combined_score = web_weight * factcheck_p_true + bert_weight * bert_p_true
        overall_confidence = (bert_conf + factcheck_conf) / 2
    
    # 4) Determine final label with improved thresholds
    if combined_score >= 0.7:
        final_label = "real"
    elif combined_score <= 0.3:
        final_label = "fake"
    else:
        final_label = "inconclusive"
    
    # 5) Adjust label based on overall confidence
    if overall_confidence < 0.6:
        final_label = "inconclusive"
    
    return {
        "combined_score": round(combined_score, 4),
        "final_label": final_label,
        "bert_score": round(bert_p_true, 4),
        "factcheck_score": round(factcheck_p_true, 4),
        "confidence": round(overall_confidence, 4),
        "bert_confidence": round(bert_conf, 4),
        "factcheck_confidence": round(factcheck_conf, 4)
    }