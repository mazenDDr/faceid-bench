"""Face detectors behind one interface: image (BGR, HxWx3 uint8) -> (n, 15) float array.

Columns: x, y, w, h, five landmarks (right eye, left eye, nose, right mouth, left mouth) as
x/y pairs, score. Landmarks feed the alignment step of the verification pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS = Path("models")


class YuNet:
    """OpenCV's FaceDetectorYN, run at the image's own size (the published evaluation setting)."""

    def __init__(
        self,
        version: str = "2023mar",
        score: float = 0.3,
        nms: float = 0.45,
        top_k: int = 5000,
    ):
        import cv2

        self.name = f"yunet_{version}"
        self.detector = cv2.FaceDetectorYN.create(
            str(MODELS / f"{self.name}.onnx"), "", (320, 320), score, nms, top_k
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        return np.zeros((0, 15), np.float32) if faces is None else faces.astype(np.float32)


class SCRFD:
    """InsightFace SCRFD (with keypoints), decoded like its published WIDER FACE test config.

    Resize keeping the aspect ratio so the long side is `size`, pad bottom/right to size x size,
    normalise (x - 127.5) / 128 in RGB, keep scores >= 0.02, NMS at IoU 0.45, no top-k.
    """

    strides = (8, 16, 32)
    anchors = 2

    def __init__(
        self,
        path: str,
        size: int = 640,
        score: float = 0.02,
        nms: float = 0.45,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ):
        import onnxruntime as ort

        self.name = Path(path).stem
        self.size, self.score, self.nms = size, score, nms
        self.session = ort.InferenceSession(str(MODELS / path), providers=list(providers))
        self.input = self.session.get_inputs()[0].name
        self.centers = {}

    def grid(self, stride: int) -> np.ndarray:
        if stride not in self.centers:
            n = self.size // stride
            ys, xs = np.mgrid[:n, :n]
            points = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            self.centers[stride] = np.repeat(points, self.anchors, axis=0)
        return self.centers[stride]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        canvas = np.zeros((self.size, self.size, 3), np.float32)
        canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        blob = ((canvas[:, :, ::-1] - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outputs = self.session.run(None, {self.input: np.ascontiguousarray(blob)})

        rows = []
        for i, stride in enumerate(self.strides):
            scores = outputs[i].reshape(-1)
            keep = scores >= self.score
            centers = self.grid(stride)[keep]
            dist = outputs[i + 3][keep] * stride
            kps = outputs[i + 6][keep] * stride
            boxes = np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], axis=1)
            kps = kps + np.tile(centers, 5)
            rows.append(np.column_stack([boxes, kps, scores[keep]]))
        det = np.concatenate(rows) if rows else np.zeros((0, 15), np.float32)
        det = det[nms(det[:, :4], det[:, 14], self.nms)]
        # back to original pixels; mmdet divides by the per-axis resize factor
        factor = np.array([new_w / w, new_h / h], np.float32)
        det[:, :14] /= np.tile(factor, 7)
        det[:, 2:4] -= det[:, :2]  # x1 y1 x2 y2 -> x y w h
        return det.astype(np.float32)


class Downscaled:
    """Shrink the image so its long side is `size`, detect, and map results back.

    Simulates a small camera frame: a close face (the Face ID case) stays large enough to find.
    """

    def __init__(self, detector, size: int):
        self.detector, self.size = detector, size
        self.name = f"{detector.name}@{size}"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        scale = self.size / max(image.shape[:2])
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        det = self.detector(small)
        det[:, :14] /= scale
        return det


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS on x1y1x2y2 boxes (no +1 offset, as mmcv's nms). Returns kept indices."""
    order = np.argsort(-scores, kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:4], boxes[rest, 2:4])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
        order = rest[inter / (areas[i] + areas[rest] - inter) <= iou]
    return np.array(keep, dtype=np.int64)


DETECTORS = {
    "yunet_2023mar": lambda: YuNet("2023mar"),
    "yunet_2026may": lambda: YuNet("2026may"),
    "scrfd_10g_kps": lambda: SCRFD("buffalo_l/det_10g.onnx"),
    "scrfd_500m_kps": lambda: SCRFD("buffalo_s/det_500m.onnx"),
    # 320 px input: the long side is 320, as a small camera frame would give
    "yunet@320": lambda: Downscaled(YuNet("2026may"), 320),
    "scrfd_10g_kps@320": lambda: SCRFD("buffalo_l/det_10g.onnx", size=320),
    "scrfd_500m_kps@320": lambda: SCRFD("buffalo_s/det_500m.onnx", size=320),
    # fp16 copies (what the Neural Engine runs), on CUDA to check accuracy is unchanged
    "scrfd_10g_kps_fp16": lambda: SCRFD(
        "fixed/scrfd_10g_kps_640_fp16.onnx", providers=("CUDAExecutionProvider",)
    ),
    "scrfd_500m_kps_fp16": lambda: SCRFD(
        "fixed/scrfd_500m_kps_640_fp16.onnx", providers=("CUDAExecutionProvider",)
    ),
}
