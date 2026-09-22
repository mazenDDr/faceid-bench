"""Laya (NandhaKishorM/laya) as the match decision engine.

Laya reads text or JSON, not pixels, so it cannot detect faces. What it can do is the last step
of the pipeline: read the numbers the vision models produce and answer a typed question with a
probability. Here it answers the `noul` question "same person?" from a JSON state, and is
compared with the two-number Platt calibration of the cosine score.
"""

from __future__ import annotations

import numpy as np

QUESTION = {
    "same_person": {
        "type": "noul",
        "instructions": "Are the two face photos of the same person?",
    }
}
NOTE = (
    "face_similarity is the cosine similarity of two face embeddings: the same person usually "
    "scores above 0.3, two different people usually score near 0."
)


def state(variant: str, sim: float, det_a: np.ndarray, det_b: np.ndarray) -> dict:
    """The JSON Laya sees for one pair. det_* are 15-column detector rows (x y w h ... score)."""
    if variant == "plain":
        return {"face_similarity": round(float(sim), 4)}
    if variant == "note":
        return {"face_similarity": round(float(sim), 4), "note": NOTE}
    if variant == "rich":
        return {
            "face_similarity": round(float(sim), 4),
            "note": NOTE,
            "photo_a": {
                "detector_confidence": round(float(det_a[14]), 3),
                "face_width_px": int(det_a[2]),
            },
            "photo_b": {
                "detector_confidence": round(float(det_b[14]), 3),
                "face_width_px": int(det_b[2]),
            },
        }
    raise ValueError(variant)


VARIANTS = ("plain", "note", "rich")


class LayaJudge:
    """`noul` answers for many states, with the same maths as Agent.system_one.

    Batching several states in one forward pass changed answers by up to 0.04 (0.07 with
    padding) against agent.predict, so the default is one state per pass, which matches
    predict to 5e-5.
    """

    def __init__(self, model_id: str = "convaiinnovations/laya", device: str = "cuda"):
        import laya

        self.agent = laya.load(model_id, device=device)

    def noul(self, states: list[dict], question: dict = QUESTION, batch_size: int = 1):
        import torch
        from laya.common import QTYPES, build_sequence, collate_items, temp_bucket

        a = self.agent
        (qdef,) = question.values()
        q = a._to_internal(qdef)
        qt = QTYPES[q["t"]]
        max_len, head = a.cfg.get("max_len", 512), a.cfg.get("head_max_len", 192)
        out = []
        for i in range(0, len(states), batch_size):
            items = []
            for s in states[i : i + batch_size]:
                seq, markers = build_sequence(a.tok, s, q, max_len, head)
                items.append({"ids": seq, "markers": markers, "qtype": qt})
            b = collate_items([items], a.tok.pad_token_id)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=a.device.type, dtype=a.dtype, enabled=a.device.type == "cuda"
                ),
            ):
                logits, _ = a.model(
                    *(
                        b[k].to(a.device)
                        for k in (
                            "input_ids",
                            "attention_mask",
                            "marker_pos",
                            "marker_mask",
                            "qtype",
                        )
                    )
                )
            logits = logits.float().cpu().numpy()
            for r, it in enumerate(items):
                k = len(it["markers"])
                t = a.temperature_by_options.get(temp_bucket(qt, k), a.temperature[qt])
                z = logits[r, :k] / t
                p = np.exp(z - z.max())
                out.append(float((p / p.sum())[1]))
        return np.array(out)
