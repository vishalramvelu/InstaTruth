from bert import run_bert_model
from websearch import multi_step_fact_check, summarize_transcript_simple
from integratedresult import combine_predictions

def evaluate_claim(transcript: str):
    """
    Enhanced fact-checking pipeline with multi-step analysis.
    
    Args:
        transcript: Text to fact-check
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    if not transcript or not transcript.strip():
        return {
            "combined_score": 0.5,
            "final_label": "inconclusive", 
            "bert_score": 0.5,
            "factcheck_score": 0.5,
            "confidence": 0.3,
            "summary": "No content provided for analysis"
        }
    
    try:
        # Step 1: BERT Analysis
        print("[1/2] Running BERT classification...")
        bert_out = run_bert_model(transcript)
        print(f"BERT result: {bert_out}")
        
    except Exception as e:
        print(f"BERT processing failed: {e}")
        # Continue with default BERT values
        bert_out = {
            "label": "inconclusive",
            "score": 0.5,
            "confidence": 0.3
        }
    
    try:
        # Step 2: Enhanced Multi-Step Web Fact-Checking
        print("[2/2] Running multi-step fact-checking...")
        factcheck_result = multi_step_fact_check(transcript)
        print(f"Fact-check result: {factcheck_result}")
        
        # Convert multi-step result to format expected by combine_predictions
        explanation = {
            "summary": factcheck_result.get("summary", ""),
            "confidence": factcheck_result.get("confidence", 0.5),
            "verdict": factcheck_result.get("verdict", "inconclusive"),
            "claims_analyzed": factcheck_result.get("claims_analyzed", 0),
            "individual_results": factcheck_result.get("individual_results", [])
        }
        
    except Exception as e:
        print(f"Multi-step fact-checking failed: {e}")
        # Fallback to basic summary
        try:
            summary = summarize_transcript_simple(transcript, 80)
            explanation = {
                "summary": f"Analysis failed, summary: {summary}",
                "confidence": 0.3,
                "verdict": "inconclusive"
            }
        except:
            explanation = {
                "summary": "Both fact-checking and summarization failed",
                "confidence": 0.2,
                "verdict": "inconclusive"
            }
    
    try:
        # Step 3: Combine BERT + Enhanced Web Analysis
        print("[3/3] Combining results...")
        final = combine_predictions(bert_out=bert_out, explanation=explanation)
        
        # Add enhanced metadata
        final['summary'] = explanation['summary']
        if 'claims_analyzed' in explanation:
            final['claims_analyzed'] = explanation['claims_analyzed']
        if 'individual_results' in explanation:
            final['individual_claims'] = explanation['individual_results']
        
        print("=== FINAL ANALYSIS ===")
        print(f"Final Label: {final['final_label']}")
        print(f"Combined Score: {final['combined_score']}")
        print(f"BERT Score: {final['bert_score']}")
        print(f"Fact-check Score: {final['factcheck_score']}")
        print(f"Summary: {final['summary']}")
        
        return final

    except Exception as e:
        print(f"Result combination failed: {e}")
        # Return fallback result
        return {
            "combined_score": 0.4,
            "final_label": "inconclusive",
            "bert_score": bert_out.get("score", 0.5),
            "factcheck_score": explanation.get("confidence", 0.5),
            "confidence": 0.3,
            "summary": f"Analysis partially failed: {str(e)[:100]}"
        }
    
    



if __name__ == "__main__":
    text = "LeBron James is one of the best basketball players of all time. He dominates in scoring, playmaking," \
"rebounding, and defense across 20 years in his career. He has led three different franchies to four championships showing" \
"how he is able to do lead very well. In Febuary 2023 he passed Kareem Abdul-Jabbar to become NBA's all time leading scorer. " \
"He is still continuing to play today for the Los Angeles Lakers and his currently teammates with Luka Doncic. " \
"He is competing for a championship next season and remains optimistic about future."
    evaluate_claim(text)