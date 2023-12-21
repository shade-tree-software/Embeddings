import sys
from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text):
    inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return summary

if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "rt") as f:
        text = f.read()
    summary = summarize_text(text)
    if summary:
        print(f"\n{summary}\n")
    else:
        print("document is invalid or is too long to summarize")
