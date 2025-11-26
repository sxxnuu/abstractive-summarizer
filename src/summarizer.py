from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AbstractiveSummarizer:
    def __init__(self, model_name="gogamza/kobart-summarizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_len=128):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
