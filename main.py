# Text summarization
# streamlit run summarizeAPP.py

import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

class TextSummarizer:

    def __init__(self):
        # Load pretrained model and tokenizer
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    def summarize(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)


        summary_ids = self.model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)


    
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True) # skip_special_tokens like padding tokens
        return summary

def main():
    st.title("Text Summarization App")
    summarizer = TextSummarizer()

    text = st.text_area("Enter text for summarization")
    if st.button("Summarize"):
        if text:
            summary = summarizer.summarize(text)
            st.write(f"Summary: {summary}")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
