# tests/test_inference.py
import tempfile
import numpy as np
import onnxruntime as ort
from app.model_builder import build_add_model


def test_add_model_inference():
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/model.onnx"
        build_add_model(path)

        sess = ort.InferenceSession(path)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        res = sess.run(None, {'A': a, 'B': b})
        assert (np.array(res[0]) == np.array([5.0, 7.0, 9.0])).all()