from bert import run_bert_model
from websearch import google_search, deepseek_reason_over_results, summarize_transcript
from transformers import pipeline
from integratedresult import combine_predictions

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def evaluate_claim(transcript: str):
    #1) run bert 
    try:
        bert_out = run_bert_model(transcript)
        # {'label': (str), 'score': (float)}
    
    except Exception as e:
        print("BERT processing failed:", e)
        return
    
    #2)run web search pipeline

    #2a)google search
    try:
        query = summarize_transcript(transcript, 80)
        results = google_search(query, num_results=5)
        # print("=== GOOGLE SEARCH RESULTS ===")
        # for idx, item in enumerate(results, start=1):
        #     print(f"{idx}. {item['title']}")
        #     print(f"   {item['displayLink']}")
        #     print(f"   {item['snippet']}")
        #     print(f"   {item['link']}\n")
    except Exception as e:
        print(f"Google search failed: {e}")
        return
    
    #2b) LLM integration
    try:
        claim = query
        factcheck = deepseek_reason_over_results(results=results, claim_text=claim, num_sources=len(results))
        # print("=== DEEPSEEK FACT-CHECK SUMMARY ===")
        # print(factcheck)

    except Exception as e:
        print(f"DeepSeek reasoning failed: {e}")
        return


    #3) Integrate BERT + Web search for final result
    try:
        final = combine_predictions(bert_out = bert_out, explanation=factcheck)
        final['summary'] = factcheck['summary']
        print(final)
        return(final)

    except Exception as e:
        print("Combine step failed:", e)
        return
    
    



if __name__ == "__main__":
    text = "LeBron James is one of the best basketball players of all time. He dominates in scoring, playmaking," \
"rebounding, and defense across 20 years in his career. He has led three different franchies to four championships showing" \
"how he is able to do lead very well. In Febuary 2023 he passed Kareem Abdul-Jabbar to become NBA's all time leading scorer. " \
"He is still continuing to play today for the Los Angeles Lakers and his currently teammates with Luka Doncic. " \
"He is competing for a championship next season and remains optimistic about future."
    evaluate_claim(text)