"""The vectorised mask helpers must reproduce the original loops bit for bit."""

import cv2
import numpy as np
import pytest

from comic_text_detector.utils import textmask
from comic_text_detector.utils.imgproc_utils import union_area
from comic_text_detector.utils.textblock import TextBlock


def _get_topk_color_original(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    if len(color_list) > 1:
        for color, bin in zip(color_list[1:], bins[1:]):
            if np.abs(np.array(top_colors) - color).min() > color_var:
                top_colors.append(color)
            if len(top_colors) >= k or bin < bin_tol:
                break
    return top_colors


def _merge_loop_original(mask_merged, pred_mask, labels, num_labels, stats, min_wh=3, area_thresh=None):
    """Original greedy loop of merge_mask_list (both variants: w*h filter / area filter)."""
    mask_merged = mask_merged.copy()
    for label_index, stat in zip(range(num_labels), stats):
        x, y, w, h, area = stat
        if area_thresh is None:
            if label_index == 0 or w * h < min_wh:
                continue
        elif not area < area_thresh:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        label_local = labels[y1:y2, x1:x2]
        label_cordinates = np.where(label_local == label_index)
        tmp_merged = np.zeros_like(label_local, np.uint8)
        tmp_merged[label_cordinates] = 255
        tmp_merged = cv2.bitwise_or(mask_merged[y1:y2, x1:x2], tmp_merged)
        xor_merged = cv2.bitwise_xor(tmp_merged, pred_mask[y1:y2, x1:x2]).sum()
        xor_origin = cv2.bitwise_xor(mask_merged[y1:y2, x1:x2], pred_mask[y1:y2, x1:x2]).sum()
        if xor_merged < xor_origin:
            mask_merged[y1:y2, x1:x2] = tmp_merged
    return mask_merged


def _refine_undetected_mask_original(img, mask_pred, mask_refined, blk_list, refine_mode=textmask.REFINEMASK_INPAINT):
    """Original refine_undetected_mask (scalar loop over blocks for every component)."""
    mask_pred = mask_pred.copy()
    mask_refined = mask_refined.copy()
    mask_pred[np.where(mask_refined > 30)] = 0
    _, pred_mask_t = cv2.threshold(mask_pred, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)
    valid_labels = np.where(stats[:, -1] > 50)[0]
    seg_blk_list = []
    if len(valid_labels) > 0:
        for lab_index in valid_labels[1:]:
            x, y, w, h, area = stats[lab_index]
            bbox = [x, y, x + w, y + h]
            bbox_score = -1
            for blk in blk_list:
                bbox_s = union_area(blk.xyxy, bbox)
                if bbox_s > bbox_score:
                    bbox_score = bbox_s
            if bbox_score / w / h < 0.5:
                seg_blk_list.append(TextBlock(bbox))
    if len(seg_blk_list) > 0:
        mask_refined = cv2.bitwise_or(
            mask_refined, textmask.refine_mask(img, mask_pred, seg_blk_list, refine_mode=refine_mode)
        )
    return mask_refined


@pytest.mark.parametrize('seed', range(20))
def test_get_topk_color(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    bins = rng.integers(0, 50, size=n).astype(np.int64)
    if seed % 4 == 0:
        bins[rng.integers(0, n, size=n // 2 + 1)] = 7  # heavy ties
    colors = np.sort(rng.random(n + 1) * 255)
    k = int(rng.integers(1, 6))
    ref = _get_topk_color_original(bins.copy(), colors[:-1].copy(), k=k)
    new = textmask.get_topk_color(bins.copy(), colors[:-1].copy(), k=k)
    assert [float(c) for c in ref] == [float(c) for c in new]


def _random_masks(rng, h=60, w=80):
    pred = (rng.random((h, w)) < 0.35).astype(np.uint8) * 255
    cand = (rng.random((h, w)) < 0.3).astype(np.uint8) * 255
    return pred, cand


@pytest.mark.parametrize('seed', range(10))
def test_merge_components_binary(seed):
    rng = np.random.default_rng(seed)
    pred, cand = _random_masks(rng)
    merged0 = (rng.random(pred.shape) < 0.05).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8, cv2.CV_16U)
    ref = _merge_loop_original(merged0, pred, labels, num, stats)
    keep = (stats[:, 2].astype(np.int64) * stats[:, 3]) >= 3
    new = textmask._merge_components(merged0.copy(), pred, labels, num, keep, binary_pred=True)
    assert np.array_equal(ref, new)


@pytest.mark.parametrize('seed', range(10))
def test_merge_components_hole_fill(seed):
    rng = np.random.default_rng(seed)
    pred, cand = _random_masks(rng)
    merged0 = cand.copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(255 - merged0, 8, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    area_thresh = sorted_area[-2] if len(sorted_area) > 1 else sorted_area[-1]
    ref = _merge_loop_original(merged0, pred, labels, num, stats, area_thresh=area_thresh)
    keep = stats[:, -1] < area_thresh
    new = textmask._merge_components(merged0.copy(), pred, labels, num, keep, binary_pred=True)
    assert np.array_equal(ref, new)


@pytest.mark.parametrize('seed', range(5))
def test_merge_components_float_weights(seed):
    """pred_thresh <= 0 path: pred_mask is not binary."""
    rng = np.random.default_rng(seed)
    pred = rng.integers(0, 256, size=(60, 80), dtype=np.uint8)
    cand = (rng.random((60, 80)) < 0.3).astype(np.uint8) * 255
    merged0 = np.zeros_like(pred)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8, cv2.CV_16U)
    ref = _merge_loop_original(merged0, pred, labels, num, stats)
    keep = (stats[:, 2].astype(np.int64) * stats[:, 3]) >= 3
    new = textmask._merge_components(merged0.copy(), pred, labels, num, keep, binary_pred=False)
    assert np.array_equal(ref, new)


@pytest.mark.parametrize('seed', range(5))
def test_refine_undetected_mask(seed):
    """Vectorised block-overlap score vs the original scalar loop."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    mask_pred = np.zeros((120, 160), np.uint8)
    cv2.rectangle(mask_pred, (10, 10), (50, 40), 255, -1)
    cv2.rectangle(mask_pred, (80, 60), (150, 110), 255, -1)
    cv2.rectangle(mask_pred, (100, 10), (130, 30), 255, -1)
    mask_refined = np.zeros_like(mask_pred)
    cv2.rectangle(mask_refined, (10, 10), (30, 30), 255, -1)
    mask_refined[rng.random(mask_refined.shape) < 0.01] = 255

    blk_list = [
        TextBlock([12, 12, 34, 36]),
        TextBlock([82, 62, 150, 110]),
        TextBlock([0, 0, 159, 20]),
        TextBlock([40, 50, 60, 70]),
    ]
    for refine_mode in (textmask.REFINEMASK_INPAINT, textmask.REFINEMASK_ANNOTATION):
        ref = _refine_undetected_mask_original(img, mask_pred, mask_refined, blk_list, refine_mode)
        new = textmask.refine_undetected_mask(img, mask_pred.copy(), mask_refined.copy(), blk_list, refine_mode)
        assert np.array_equal(ref, new)
