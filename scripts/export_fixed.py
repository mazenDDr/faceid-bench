"""Write fixed-shape (and fp16) copies of the detectors for latency timing.

Core ML and TensorRT need static shapes, and the Neural Engine only runs fp16 (T02), so each
detector gets models/fixed/<name>_<size>.onnx and <name>_<size>_fp16.onnx.
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

SOURCES = {
    "yunet": "models/yunet_2026may.onnx",
    "scrfd_500m_kps": "models/buffalo_s/det_500m.onnx",
    "scrfd_10g_kps": "models/buffalo_l/det_10g.onnx",
}
RECOGNIZERS = {
    "mbf_w600k": "models/buffalo_s/w600k_mbf.onnx",
    "r50_w600k": "models/buffalo_l/w600k_r50.onnx",
    "sface": "models/sface_2021dec.onnx",
}
out_dir = Path("models/fixed")
out_dir.mkdir(parents=True, exist_ok=True)


def fix_input(model: onnx.ModelProto, shape: list[int]) -> onnx.ModelProto:
    """Fix the input shape and re-derive every other shape from it.

    Output shapes stored in the file were computed for the original input (SCRFD ships with
    640x640 values), so they are cleared first; otherwise a 320 export keeps 640 output shapes.
    """
    make_input_shape_fixed(model.graph, model.graph.input[0].name, shape)
    del model.graph.value_info[:]
    for output in model.graph.output:
        output.type.tensor_type.ClearField("shape")
    # shape inference cannot follow SCRFD's computed reshapes, so read the real output shapes
    # from one run and store them as static dims
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {session.get_inputs()[0].name: np.zeros(shape, np.float32)})
    for output, value in zip(model.graph.output, outputs, strict=True):
        dims = output.type.tensor_type.shape.dim
        for size in value.shape:
            dims.add().dim_value = size
    return onnx.shape_inference.infer_shapes(model)


for name, source in SOURCES.items():
    for size in (640, 320):
        model = fix_input(onnx.load(source), [1, 3, size, size])
        onnx.save(model, out_dir / f"{name}_{size}.onnx")
        fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(fp16, out_dir / f"{name}_{size}_fp16.onnx")
        print(f"{name}_{size}: fp32 + fp16")

# Recognizers: batch 1 at 112x112 for timing; an fp16 copy keeping the dynamic batch for accuracy.
for name, source in RECOGNIZERS.items():
    model = fix_input(onnx.load(source), [1, 3, 112, 112])
    onnx.save(model, out_dir / f"{name}_112.onnx")
    onnx.save(
        float16.convert_float_to_float16(model, keep_io_types=True),
        out_dir / f"{name}_112_fp16.onnx",
    )
    dynamic = float16.convert_float_to_float16(onnx.load(source), keep_io_types=True)
    onnx.save(dynamic, out_dir / f"{name}_fp16.onnx")
    print(f"{name}_112: fp32 + fp16, plus dynamic-batch fp16")
