import os
from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

# Put your .gguf file path here OR set it via MODEL_PATH env var
MODEL_PATH = os.environ.get("MODEL_PATH", "SmolLM2-360M-Instructt-q8_0.gguf")

# Load model once at startup
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # context size
    n_threads=8,       # change based on CPU
    n_gpu_layers=0,    # 0 = CPU only (set >0 if you built with GPU support)
    verbose=False
)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH)})

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please provide 'question' in JSON body"}), 400

    # Minimal prompt format
    prompt = f"User: {question}\nAssistant:"

    out = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
        stop=["User:"]
    )

    answer = out["choices"][0]["text"].strip()
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    # Run: python app.py
    app.run(host="0.0.0.0", port=8000, debug=False)
