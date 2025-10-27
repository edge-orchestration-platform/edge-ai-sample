# app/main.py
import os
import numpy as np
from flask import Flask, request, jsonify
import onnxruntime as ort
from pathlib import Path
from .model_builder import build_add_model

MODEL_PATH = Path(__file__).parent / 'model.onnx'

app = Flask(__name__)

# Ensure model exists
if not MODEL_PATH.exists():
    build_add_model(str(MODEL_PATH))

# Create session
sess = ort.InferenceSession(str(MODEL_PATH))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/infer', methods=['POST'])
def infer():
    payload = request.get_json(force=True)
    a = np.array(payload.get('a', []), dtype=np.float32)
    b = np.array(payload.get('b', []), dtype=np.float32)

    if a.shape != b.shape:
        return jsonify({'error': 'shapes of a and b must match'}), 400

    # ONNX expects inputs with names as in the model: 'A' and 'B'
    inputs = {
        'A': a,
        'B': b,
    }
    outputs = sess.run(None, inputs)
    # outputs is a list; the first output is C
    c = outputs[0].tolist()
    return jsonify({'c': c})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))