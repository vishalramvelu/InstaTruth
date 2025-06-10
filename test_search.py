from websearch import google_search, deepseek_reason_over_results, summarize_transcript
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def main(transcript: str):

    query = summarize_transcript(transcript, 80)
    #print(query)

    # 1) Use google_search() to fetch raw snippets
    try:
        results = google_search(query, num_results=5)
        print("=== GOOGLE SEARCH RESULTS ===")
        for idx, item in enumerate(results, start=1):
            print(f"{idx}. {item['title']}")
            print(f"   {item['displayLink']}")
            print(f"   {item['snippet']}")
            print(f"   {item['link']}\n")
    except Exception as e:
        print(f"Google search failed: {e}")
        return

    # 2) Feed those same results into deepseek_reason_over_results()
    claim = query
    try:
        factcheck = deepseek_reason_over_results(
            results=results,
            claim_text=claim,
            num_sources=len(results)
        )
        print("=== DEEPSEEK FACT-CHECK SUMMARY ===")
        print(f"Summary: {factcheck['summary']}")
        print(f"Confidence: {factcheck['confidence']}")
        print(f"Verdict: {factcheck['verdict']}")
    except Exception as e:
        print(f"DeepSeek reasoning failed: {e}")

if __name__ == "__main__":
    example_text = "LeBron James is one of the best basketball players of all time. He dominates in scoring, playmaking," \
"rebounding, and defense across 20 years in his career. He has led three different franchies to four championships showing" \
"how he is able to do lead very well. In Febuary 2023 he passed Kareem Abdul-Jabbar to become NBA's all time leading scorer. " \
"He is still continuing to play today for the Los Angeles Lakers and his currently teammates with Luka Doncic. " \
"He is competing for a championship next season and remains optimistic about future."
    main(example_text)
