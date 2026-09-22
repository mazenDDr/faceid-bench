import numpy as np
import pytest

from faceid_bench.laya_decision import NOTE, QUESTION, VARIANTS, state


def test_states_carry_the_similarity_and_only_the_intended_extras():
    det = np.zeros(15)
    det[2], det[14] = 88.6, 0.91234
    plain, note, rich = (state(v, 0.412345, det, det) for v in VARIANTS)
    assert plain == {"face_similarity": 0.4123}
    assert note == {"face_similarity": 0.4123, "note": NOTE}
    assert rich["photo_a"] == {"detector_confidence": 0.912, "face_width_px": 88}
    assert QUESTION["same_person"]["type"] == "noul"
    with pytest.raises(ValueError):
        state("unknown", 0.1, det, det)
