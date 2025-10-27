import os
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

app = Flask(__name__)

def summarize_text(text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text, "parameters": {"min_length": 30, "max_length": 130}}
    
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        out = resp.json()
        if isinstance(out, list) and out:
            return out[0]["summary_text"]
        return str(out)
    except Exception as e:
        return f"Hugging Face API inference failed: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    text = ""
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            summary = summarize_text(text)
    return render_template("index.html", text=text, summary=summary)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)

