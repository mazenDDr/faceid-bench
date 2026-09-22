"""Face embedding models behind one interface: (image BGR, detection row) -> unit vector.

The detection row is the 15-column detector output (x, y, w, h, 10 landmark values, score).
"""

from __future__ import annotations

import numpy as np

from faceid_bench.align import align
from faceid_bench.detectors import MODELS


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class ArcFaceONNX:
    """InsightFace recognition models (WebFace600K). Normalisation is not in the graph, so
    inputs are RGB, (x - 127.5) / 127.5, as InsightFace's loader does for these files."""

    def __init__(self, path: str, providers: tuple[str, ...] = ("CPUExecutionProvider",)):
        import onnxruntime as ort

        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        self.session = ort.InferenceSession(str(MODELS / path), providers=list(providers))
        self.input = self.session.get_inputs()[0].name

    def crops_to_blob(self, crops: np.ndarray) -> np.ndarray:
        rgb = crops[..., ::-1].astype(np.float32)
        return np.ascontiguousarray(((rgb - 127.5) / 127.5).transpose(0, 3, 1, 2))

    def embed_crops(self, crops: np.ndarray) -> np.ndarray:
        return normalize(self.session.run(None, {self.input: self.crops_to_blob(crops)})[0])

    def __call__(self, image: np.ndarray, det: np.ndarray) -> np.ndarray:
        return self.embed_crops(align(image, det[4:14])[None])[0]


class SFace:
    """OpenCV Zoo SFace through cv2.FaceRecognizerSF (its own alignment and normalisation)."""

    def __init__(self, path: str = "sface_2021dec.onnx"):
        import cv2

        self.model = cv2.FaceRecognizerSF.create(str(MODELS / path), "")

    def __call__(self, image: np.ndarray, det: np.ndarray) -> np.ndarray:
        crop = self.model.alignCrop(image, det.reshape(1, -1).astype(np.float32))
        return normalize(self.model.feature(crop).reshape(-1))


EMBEDDERS = {
    "mbf_w600k": lambda providers: ArcFaceONNX("buffalo_s/w600k_mbf.onnx", providers),
    "r50_w600k": lambda providers: ArcFaceONNX("buffalo_l/w600k_r50.onnx", providers),
    "sface": lambda providers: SFace(),
    # fp16 copies (what the Neural Engine runs), to check accuracy is unchanged
    "mbf_w600k_fp16": lambda providers: ArcFaceONNX("fixed/mbf_w600k_fp16.onnx", providers),
    "r50_w600k_fp16": lambda providers: ArcFaceONNX("fixed/r50_w600k_fp16.onnx", providers),
}
