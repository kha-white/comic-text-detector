from typing import List

import cv2
import numpy as np

from .imgproc_utils import expand_textwindow, union_area
from .textblock import TextBlock

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LANG_ENG = 0
LANG_JPN = 1

REFINEMASK_INPAINT = 0
REFINEMASK_ANNOTATION = 1

def get_topk_color(color_list, bins, k=3, color_var=10, bin_tol=0.001):
    idx = np.argsort(bins * -1)
    color_list, bins = color_list[idx], bins[idx]
    top_colors = [color_list[0]]
    bin_tol = np.sum(bins) * bin_tol
    # Closed form of the original loop: consider colours up to the first one
    # whose bin is below bin_tol, pick up to k that are color_var apart.
    n = len(color_list)
    if n > 1:
        below = np.flatnonzero(bins[1:] < bin_tol)
        last = below[0] + 1 if len(below) else n - 1
        if k <= 1:
            last = 1  # the original loop always ran its first iteration
            k = 2
        cols = color_list[1:last + 1]
        ok = np.abs(cols - top_colors[0]) > color_var
        start = 0
        while len(top_colors) < k:
            cand = np.flatnonzero(ok[start:])
            if len(cand) == 0:
                break
            j = start + cand[0]
            c = cols[j]
            top_colors.append(c)
            ok &= np.abs(cols - c) > color_var
            start = j + 1
    return top_colors

def minxor_thresh(threshed, mask, dilate=False):
    neg_threshed = 255 - threshed
    e_size = 1
    if dilate:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
        neg_threshed = cv2.dilate(neg_threshed, element, iterations=1)
        threshed = cv2.dilate(threshed, element, iterations=1)
    neg_xor_sum = cv2.bitwise_xor(neg_threshed, mask).sum()
    xor_sum = cv2.bitwise_xor(threshed, mask).sum()
    if neg_xor_sum < xor_sum:
        return neg_threshed, neg_xor_sum
    else:
        return threshed, xor_sum

def get_otsuthresh_masklist(img, pred_mask, per_channel=False) -> List[np.ndarray]:
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    mask_list = []
    for c in channels:
        _, threshed = cv2.threshold(c, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        threshed, xor_sum = minxor_thresh(threshed, pred_mask, dilate=False)
        mask_list.append([threshed, xor_sum])
    mask_list.sort(key=lambda x: x[1])
    if per_channel:
        return mask_list
    else:
        return [mask_list[0]]

def get_topk_masklist(im_grey, pred_mask):
    if len(im_grey.shape) == 3 and im_grey.shape[-1] == 3:
        im_grey = cv2.cvtColor(im_grey, cv2.COLOR_BGR2GRAY)
    msk = np.ascontiguousarray(pred_mask)
    candidate_grey_px = im_grey[cv2.erode(msk, np.ones((3,3), np.uint8), iterations=1) > 127]
    bin, his = np.histogram(candidate_grey_px, bins=255)
    topk_color = get_topk_color(his, bin, color_var=10, k=3)
    color_range = 30
    mask_list = list()
    for ii, color in enumerate(topk_color):
        c_top = min(color+color_range, 255)
        c_bottom = c_top - 2 * color_range
        threshed = cv2.inRange(im_grey, c_bottom, c_top)
        threshed, xor_sum = minxor_thresh(threshed, msk)
        mask_list.append([threshed, xor_sum])
    return mask_list

def _merge_components(mask_merged, pred_mask, labels, num_labels, keep_candidates, binary_pred=False):
    """Vectorised form of the per-component merge loop in merge_mask_list."""
    # Accepting a component only changes its pixels where mask_merged is 0, each
    # by (255 ^ p) - p; components are pixel-disjoint, so every accept/reject
    # decision can be summed in one np.bincount pass.
    if binary_pred:
        # pred_mask is exactly 0/255 here: count p==255 against p==0 per label.
        lab2 = labels.astype(np.intp)
        lab2 *= 2
        lab2 += (pred_mask != 0)
        if mask_merged.any():
            lab2[mask_merged != 0] = 0
        cnt = np.bincount(lab2.ravel(), minlength=2 * num_labels)
        keep = (cnt[1:2 * num_labels:2] > cnt[0:2 * num_labels:2]) & keep_candidates
    else:
        delta = 255 - 2 * pred_mask.astype(np.int32)
        delta[mask_merged != 0] = 0
        score = np.bincount(labels.ravel(), weights=delta.ravel(), minlength=num_labels)
        keep = (score[:num_labels] < 0) & keep_candidates
    keep[0] = False  # background label
    if keep.any():
        mask_merged[keep[labels]] = 255
    return mask_merged


def merge_mask_list(mask_list, pred_mask, blk: TextBlock = None, pred_thresh=30, text_window=None, filter_with_lines=False, refine_mode=REFINEMASK_INPAINT):
    mask_list.sort(key=lambda x: x[1])
    linemask = None
    if blk is not None and filter_with_lines:
        linemask = np.zeros_like(pred_mask)
        lines = blk.lines_array(dtype=np.int64)
        for line in lines:
            line[..., 0] -= text_window[0]
            line[..., 1] -= text_window[1]
            cv2.fillPoly(linemask, [line], 255)
        linemask = cv2.dilate(linemask, np.ones((3, 3), np.uint8), iterations=3)
    
    if pred_thresh > 0:
        e_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))      
        pred_mask = cv2.erode(pred_mask, element, iterations=1)
        _, pred_mask = cv2.threshold(pred_mask, 60, 255, cv2.THRESH_BINARY)
    binary_pred = pred_thresh > 0  # thresholded above -> exactly 0/255
    connectivity = 8
    mask_merged = np.zeros_like(pred_mask)
    for ii, (candidate_mask, xor_sum) in enumerate(mask_list):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity, cv2.CV_16U)
        if num_labels <= 1:
            continue
        # original loop skipped components whose bounding box has w*h < 3
        keep_candidates = (stats[:, 2].astype(np.int64) * stats[:, 3]) >= 3
        mask_merged = _merge_components(mask_merged, pred_mask, labels, num_labels, keep_candidates, binary_pred)

    if refine_mode == REFINEMASK_INPAINT:
        mask_merged = cv2.dilate(mask_merged, np.ones((3, 3), np.uint8), iterations=1)
    # fill holes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-mask_merged, connectivity, cv2.CV_16U)
    sorted_area = np.sort(stats[:, -1])
    if len(sorted_area) > 1:
        area_thresh = sorted_area[-2]
    else:
        area_thresh = sorted_area[-1]
    # The original loop considered every label; label 0 is the merged
    # foreground, whose delta is 0.
    keep_candidates = stats[:, -1] < area_thresh
    mask_merged = _merge_components(mask_merged, pred_mask, labels, num_labels, keep_candidates, binary_pred)
    return mask_merged


def refine_undetected_mask(img: np.ndarray, mask_pred: np.ndarray, mask_refined: np.ndarray, blk_list: List[TextBlock], refine_mode=REFINEMASK_INPAINT):
    # Boolean assignment avoids the int64 index arrays np.where built.
    mask_pred[mask_refined > 30] = 0
    _, pred_mask_t = cv2.threshold(mask_pred, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask_t, 4, cv2.CV_16U)
    valid_labels = np.where(stats[:, -1] > 50)[0]
    seg_blk_list = []
    if len(valid_labels) > 0:
        valid_labels = valid_labels[1:]
        if len(valid_labels) > 0:
            vs = stats[valid_labels]
            bx1 = vs[:, 0]; by1 = vs[:, 1]; w = vs[:, 2]; h = vs[:, 3]
            bx2 = bx1 + w; by2 = by1 + h
            # vectorised max over blk_list of union_area(blk.xyxy, bbox)
            bbox_score = np.full(len(valid_labels), -1, dtype=np.int64)
            if len(blk_list) > 0:
                bl = np.array([blk.xyxy for blk in blk_list], dtype=np.int64)  # (M, 4)
                ix1 = np.maximum(bl[:, 0][:, None], bx1[None, :])
                iy1 = np.maximum(bl[:, 1][:, None], by1[None, :])
                ix2 = np.minimum(bl[:, 2][:, None], bx2[None, :])
                iy2 = np.minimum(bl[:, 3][:, None], by2[None, :])
                inter = np.where((iy2 < iy1) | (ix2 < ix1), -1, (iy2 - iy1) * (ix2 - ix1))
                bbox_score = np.maximum(bbox_score, inter.max(axis=0))
            for i, lab_index in enumerate(valid_labels):
                x, y, w_, h_, area = stats[lab_index]
                bbox = [x, y, x+w_, y+h_]
                if bbox_score[i] / w_ / h_ < 0.5:
                    seg_blk_list.append(TextBlock(bbox))
    if len(seg_blk_list) > 0:
        mask_refined = cv2.bitwise_or(mask_refined, refine_mask(img, mask_pred, seg_blk_list, refine_mode=refine_mode))
    return mask_refined


def refine_mask(img: np.ndarray, pred_mask: np.ndarray, blk_list: List[TextBlock], refine_mode: int = REFINEMASK_INPAINT) -> np.ndarray:
    mask_refined = np.zeros_like(pred_mask)
    for blk in blk_list:
        bx1, by1, bx2, by2 = expand_textwindow(img.shape, blk.xyxy, expand_r=16)
        im = np.ascontiguousarray(img[by1: by2, bx1: bx2])
        msk = np.ascontiguousarray(pred_mask[by1: by2, bx1: bx2])
        mask_list = get_topk_masklist(im, msk)
        mask_list += get_otsuthresh_masklist(im, msk, per_channel=False)
        mask_merged = merge_mask_list(mask_list, msk, blk=blk, text_window=[bx1, by1, bx2, by2], refine_mode=refine_mode)
        mask_refined[by1: by2, bx1: bx2] = cv2.bitwise_or(mask_refined[by1: by2, bx1: bx2], mask_merged)
    return mask_refined

# def extract_textballoon(img, pred_textmsk=None, global_mask=None):
#     if len(img.shape) > 2 and img.shape[2] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     im_h, im_w = img.shape[0], img.shape[1]
#     hyp_textmsk = np.zeros((im_h, im_w), np.uint8)
#     thresh_val, threshed = cv2.threshold(img, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
#     xormap_sum = cv2.bitwise_xor(threshed, pred_textmsk).sum()
#     neg_threshed = 255 - threshed
#     neg_xormap_sum = cv2.bitwise_xor(neg_threshed, pred_textmsk).sum()
#     neg_thresh = neg_xormap_sum < xormap_sum
#     if neg_thresh:
#         threshed = neg_threshed
#     thresh_info = {'thresh_val': thresh_val,'neg_thresh': neg_thresh}
#     connectivity = 8
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshed, connectivity, cv2.CV_16U)
#     label_unchanged = np.copy(labels)
#     if global_mask is not None:
#         labels[np.where(global_mask==0)] = 0
#     text_labels = []
#     if pred_textmsk is not None:
#         text_score_thresh = 0.5
#         textbbox_map = np.zeros_like(pred_textmsk)
#         for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
#             if label_index != 0: # skip background label
#                 x, y, w, h, area = stat
#                 area *= 255
#                 x1, y1, x2, y2 = x, y, x+w, y+h
#                 label_local = labels[y1: y2, x1: x2]
#                 label_cordinates = np.where(label_local==label_index)
#                 tmp_merged = np.zeros((h, w), np.uint8)
#                 tmp_merged[label_cordinates] = 255
#                 andmap = cv2.bitwise_and(tmp_merged, pred_textmsk[y1: y2, x1: x2])
#                 text_score = andmap.sum() / area
#                 if text_score > text_score_thresh:
#                     text_labels.append(label_index)
#                     hyp_textmsk[y1: y2, x1: x2][label_cordinates] = 255
#     labels = label_unchanged
#     bubble_msk = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#     bubble_msk[np.where(labels==0)] = 255
#     # if lang == LANG_JPN:
#     bubble_msk = cv2.erode(bubble_msk, (3, 3), iterations=1)
#     line_thickness = 2
#     cv2.rectangle(bubble_msk, (0, 0), (im_w, im_h), BLACK, line_thickness, cv2.LINE_8)
#     contours, hiers = cv2.findContours(bubble_msk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

#     brect_area_thresh = im_h * im_w * 0.4
#     min_brect_area = np.inf
#     ballon_index = -1
#     maxium_pixsum = -1
#     for ii, contour in enumerate(contours):
#         brect = cv2.boundingRect(contours[ii])
#         brect_area = brect[2] * brect[3]
#         if brect_area > brect_area_thresh and brect_area < min_brect_area:
#             tmp_ballonmsk = np.zeros_like(bubble_msk)
#             tmp_ballonmsk = cv2.drawContours(tmp_ballonmsk, contours, ii, WHITE, cv2.FILLED)
#             andmap_sum = cv2.bitwise_and(tmp_ballonmsk, hyp_textmsk).sum()
#             if andmap_sum > maxium_pixsum:
#                 maxium_pixsum = andmap_sum
#                 min_brect_area = brect_area
#                 ballon_index = ii
#     if ballon_index != -1:
#         bubble_msk = np.zeros_like(bubble_msk)
#         bubble_msk = cv2.drawContours(bubble_msk, contours, ballon_index, WHITE, cv2.FILLED)
#     hyp_textmsk = cv2.bitwise_and(hyp_textmsk, bubble_msk)
#     return hyp_textmsk, bubble_msk, thresh_info, (num_labels, label_unchanged, stats, centroids, text_labels)

# def extract_textballoon_channelwise(img, pred_textmsk, test_grey=True, global_mask=None):
#     c_list = [img[:, :, i] for i in range(3)]
#     if test_grey:
#         c_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     best_xorpix_sum = np.inf
#     best_cindex = best_hyptextmsk = best_bubblemsk = best_thresh_info = best_component_stats = None
#     for c_index, channel in enumerate(c_list):
#         hyp_textmsk, bubble_msk, thresh_info, component_stats = extract_textballoon(channel, pred_textmsk, global_mask=global_mask)
#         pixor_sum = cv2.bitwise_xor(hyp_textmsk, pred_textmsk).sum()
#         if pixor_sum < best_xorpix_sum:
#             best_xorpix_sum = pixor_sum
#             best_cindex = c_index
#             best_hyptextmsk, best_bubblemsk, best_thresh_info, best_component_stats = hyp_textmsk, bubble_msk, thresh_info, component_stats
#     return best_hyptextmsk, best_bubblemsk, best_component_stats

# def refine_textmask(img, pred_mask, channel_wise=True, find_leaveouts=True, global_mask=None):
#     hyp_textmsk, bubble_msk, component_stats = extract_textballoon_channelwise(img, pred_mask, global_mask=global_mask)
#     num_labels, labels, stats, centroids, text_labels = component_stats
#     stats = np.array(stats)
#     text_stats = stats[text_labels]
#     if find_leaveouts and len(text_stats) > 0:
#         median_h = np.median(text_stats[:, 3])
#         for label, label_h in zip(range(num_labels), stats[:, 3]):
#             if label == 0 or label in text_labels:
#                 continue
#             if label_h > 0.5 * median_h and label_h < 1.5 * median_h:
#                 hyp_textmsk[np.where(labels==label)] = 255
#         hyp_textmsk = cv2.bitwise_and(hyp_textmsk, bubble_msk)
#         if global_mask is not None:
#             hyp_textmsk = cv2.bitwise_and(hyp_textmsk, global_mask)
#     return hyp_textmsk, bubble_msk