"""Write fixed-shape (and fp16) copies of the detectors for latency timing.

Core ML and TensorRT need static shapes, and the Neural Engine only runs fp16 (T02), so each
detector gets models/fixed/<name>_<size>.onnx and <name>_<size>_fp16.onnx.
"""

from pathlib import Path

import onnx
from onnxconverter_common import float16
from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

SOURCES = {
    "yunet": "models/yunet_2026may.onnx",
    "scrfd_500m_kps": "models/buffalo_s/det_500m.onnx",
    "scrfd_10g_kps": "models/buffalo_l/det_10g.onnx",
}
out_dir = Path("models/fixed")
out_dir.mkdir(parents=True, exist_ok=True)
for name, source in SOURCES.items():
    for size in (640, 320):
        model = onnx.load(source)
        make_input_shape_fixed(model.graph, model.graph.input[0].name, [1, 3, size, size])
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, out_dir / f"{name}_{size}.onnx")
        fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(fp16, out_dir / f"{name}_{size}_fp16.onnx")
        print(f"{name}_{size}: fp32 + fp16")
