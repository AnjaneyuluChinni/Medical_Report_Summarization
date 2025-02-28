import os
from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# ✅ Load LOCAL model & tokenizer
MODEL_PATH = "medical_summary_model"  # Ensure this is uploaded to GitHub
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def summarize_text(text):
    """Generates a summary for the input medical text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    summary = summarize_text(data["text"])
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)