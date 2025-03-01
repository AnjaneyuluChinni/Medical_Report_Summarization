import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ✅ Load model & tokenizer
MODEL_PATH = "medical_summary_model"  # Ensure this is uploaded
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def summarize_text(text):
    """Generates a summary for the input medical text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Create Gradio Interface
demo = gr.Interface(
    fn=summarize_text, 
    inputs=gr.Textbox(lines=5, placeholder="Enter medical text here..."), 
    outputs=gr.Textbox(),
    title="Medical Report Summarization",
    description="Enter medical text and get a summarized version using T5."
)

# ✅ Launch Gradio App
if __name__ == "__main__":

    demo.launch(share=True)
