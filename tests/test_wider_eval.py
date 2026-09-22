"""The vectorised WIDER FACE evaluator must agree with a plain loop port of the reference code."""

import numpy as np
import pytest

from faceid_bench.wider_eval import bbox_overlaps, evaluate, image_eval, image_pr, voc_ap


def reference_image_eval(pred, gt, ignore, iou):
    _pred, _gt = pred.copy(), gt.copy()
    pred_recall, recall_list, proposal_list = (
        np.zeros(len(_pred)),
        np.zeros(len(_gt)),
        np.ones(len(_pred)),
    )
    _pred[:, 2] += _pred[:, 0]
    _pred[:, 3] += _pred[:, 1]
    _gt[:, 2] += _gt[:, 0]
    _gt[:, 3] += _gt[:, 1]
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    for h in range(len(_pred)):
        max_overlap, max_idx = overlaps[h].max(), overlaps[h].argmax()
        if max_overlap >= iou:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        pred_recall[h] = len(np.where(recall_list == 1)[0])
    return pred_recall, proposal_list


def reference_img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2))
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index):
            r_index = r_index[-1]
            pr_info[t, 0] = len(np.where(proposal_list[: r_index + 1] == 1)[0])
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def reference_voc_ap(rec, prec):
    mrec, mpre = np.concatenate(([0.0], rec, [1.0])), np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def random_image(rng, n_gt=8, n_pred=15):
    gt = np.column_stack([rng.uniform(0, 200, (n_gt, 2)), rng.uniform(5, 60, (n_gt, 2))]).round()
    # half the predictions jitter a real face, half are random
    near = gt[rng.integers(0, n_gt, n_pred // 2)] + rng.normal(0, 3, (n_pred // 2, 4))
    far = np.column_stack(
        [rng.uniform(0, 200, (n_pred - len(near), 2)), rng.uniform(5, 60, (n_pred - len(near), 2))]
    )
    boxes = np.vstack([near, far])
    scores = np.sort(rng.uniform(0, 1, n_pred))[::-1]
    keep = (rng.uniform(size=n_gt) < 0.6).astype(float)
    return np.column_stack([boxes, scores]), gt, keep


@pytest.mark.parametrize("seed", range(20))
def test_image_eval_and_pr_match_reference(seed):
    rng = np.random.default_rng(seed)
    pred, gt, keep = random_image(rng)
    ours = image_eval(pred, gt, keep)
    ref = reference_image_eval(pred, gt, keep, 0.5)
    np.testing.assert_array_equal(ours[0], ref[0])
    np.testing.assert_array_equal(ours[1], ref[1])
    np.testing.assert_array_equal(
        image_pr(pred[:, 4], ours[1], ours[0]), reference_img_pr_info(1000, pred, ref[1], ref[0])
    )


@pytest.mark.parametrize("seed", range(10))
def test_voc_ap_matches_reference(seed):
    rng = np.random.default_rng(seed)
    rec, prec = np.sort(rng.uniform(size=50)), rng.uniform(size=50)
    assert voc_ap(rec, prec) == pytest.approx(reference_voc_ap(rec, prec))


def tiny_gt():
    boxes = {
        "e/a": np.array([[10, 10, 50, 50], [100, 100, 40, 40], [200, 20, 8, 8]], float),
        "e/b": np.array([[30, 30, 60, 60]], float),
    }
    subsets = {
        "easy": {"e/a": np.array([1, 2]), "e/b": np.array([1])},
        "medium": {"e/a": np.array([1, 2]), "e/b": np.array([1])},
        "hard": {"e/a": np.array([1, 2, 3]), "e/b": np.array([1])},
    }
    return {"boxes": boxes, "subsets": subsets}


def test_ground_truth_as_predictions_scores_one():
    gt = tiny_gt()
    preds = {k: np.column_stack([v, np.linspace(0.9, 0.5, len(v))]) for k, v in gt["boxes"].items()}
    aps = evaluate(preds, gt)
    assert aps == {"easy": 1.0, "medium": 1.0, "hard": 1.0}


def test_detection_on_ignored_face_is_not_a_false_positive():
    gt = tiny_gt()
    # the tiny third face is outside easy; give its detection the top score so that, if it
    # were counted as a false positive, easy precision at the first threshold would be 0
    scores = {"e/a": [0.6, 0.5, 0.95], "e/b": [0.4]}
    preds = {k: np.column_stack([v, scores[k]]) for k, v in gt["boxes"].items()}
    assert evaluate(preds, gt)["easy"] == 1.0
    preds["e/c"] = np.array([[0, 0, 5, 5, 0.99]])  # extra key is ignored by design
    assert evaluate(preds, gt)["easy"] == 1.0


def test_false_positive_lowers_ap():
    gt = tiny_gt()
    preds = {k: np.column_stack([v, np.linspace(0.9, 0.5, len(v))]) for k, v in gt["boxes"].items()}
    preds["e/b"] = np.vstack([[[300, 300, 20, 20, 0.99]], preds["e/b"]])
    assert evaluate(preds, gt)["easy"] < 1.0


def test_missing_image_raises():
    gt = tiny_gt()
    with pytest.raises(ValueError, match="no prediction entry"):
        evaluate({"e/a": np.zeros((0, 5))}, gt)


def test_nms_keeps_best_and_drops_overlaps():
    from faceid_bench.detectors import nms

    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45).tolist() == [0, 2]
    assert nms(boxes, np.array([0.7, 0.8, 0.9]), 0.45).tolist() == [2, 1]


def test_bootstrap_with_unit_weights_equals_point_estimate():
    from faceid_bench.wider_eval import ap_from_curves, bootstrap_ap, image_curves

    gt = tiny_gt()
    preds = {k: np.column_stack([v, np.linspace(0.9, 0.5, len(v))]) for k, v in gt["boxes"].items()}
    preds["e/b"] = np.vstack([[[300, 300, 20, 20, 0.99]], preds["e/b"]])
    curves, faces = image_curves(preds, gt)["easy"]
    point = ap_from_curves(curves.sum(axis=0), faces.sum())
    assert bootstrap_ap(curves, faces, np.ones((3, 2))) == pytest.approx([point] * 3)
