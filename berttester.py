from bert import run_bert_model

def main():
    # 1) A short “fake information”–style sentence
    fake_text = (
        "BREAKING: Scientists Just Discovered That Drinking Lemon Water Cures Alzheimer’s in 3 Days!"
        "According to secret research leaked from a top‐secret lab, just one glass every morning completely" 
        "reverses memory loss—even in advanced cases. Tell everyone so they can benefit from this!"
    )
    out1 = run_bert_model(fake_text)
    print("Input:", fake_text)
    print("Output:", out1)
    print("-" * 40)

    # 2) A short “real information”–style sentence
    real_text = (
        "All known living organisms are composed of cells, which are the fundamental structural"
        "and functional units of life. Whether unicellular (e.g., bacteria) or multicellular"
        "(e.g., plants and animals), every living entity relies on cellular processes to survive."
    )
    out2 = run_bert_model(real_text)
    print("Input:", real_text)
    print("Output:", out2)
    print("-" * 40)

    # 3) A neutral / debatable text block
    neutral_text = (
        "Current college admissions heavily favor applicants with access to expensive test prep "
        "and extracurriculars, perpetuating inequality. Relaxing standardized‐test requirements and "
        "valuing life experience would create a more diverse and fairly represented student body."
    )
    out3 = run_bert_model(neutral_text)
    print("Input:", neutral_text)
    print("Output:", out3)
    print("-" * 40)

if __name__ == "__main__":
    main()
