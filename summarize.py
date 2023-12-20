from vertexai.preview.language_models import TextGenerationModel
from google.api_core import exceptions
import sys

# Summarization Example with a Large Language Model

def text_summarization(text): 
    parameters = {
        "temperature": 0.2,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    try:
        return model.predict(f"Summarize the following text: {text}", **parameters).text
    except exceptions.InvalidArgument:
        return None

if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "rt") as f:
        text = f.read()
    summary = text_summarization(text)
    if summary:
        print(f"\n{summary}\n")
    else:
        print("document is invalid or is too long to summarize")