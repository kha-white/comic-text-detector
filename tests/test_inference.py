"""Bit-exactness of the refactored detector path, checked without model weights."""

import cv2
import numpy as np
import pytest
import torch

from comic_text_detector.inference import (
    TextDetector,
    letterbox_input,
    mask_to_uint8,
    normalize_input,
    postprocess_mask,
    postprocess_yolo,
    preprocess_img,
)
from comic_text_detector.utils.db_utils import SegDetectorRepresenter
from comic_text_detector.utils.imgproc_utils import letterbox
from comic_text_detector.utils.textblock import group_output
from comic_text_detector.utils.textmask import REFINEMASK_INPAINT, refine_mask, refine_undetected_mask
from comic_text_detector.utils.yolov5_utils import non_max_suppression


def _original_preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)


def _original_postprocess_mask(img, thresh=None):
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    return img.astype(np.uint8)


def _original_postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)
    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs


def _original_call(detector, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False):
    """The old TextDetector.__call__, verbatim."""
    img_in, ratio, dw, dh = _original_preprocess_img(
        img, input_size=detector.input_size, device=detector.device, half=detector.half,
        to_tensor=detector.backend == 'torch',
    )
    im_h, im_w = img.shape[:2]
    blks, mask, lines_map = detector.net(img_in)
    resize_ratio = (im_w / (detector.input_size[0] - dw), im_h / (detector.input_size[1] - dh))
    blks = _original_postprocess_yolo(blks, detector.conf_thresh, detector.nms_thresh, resize_ratio)
    if detector.backend == 'opencv':
        if mask.shape[1] == 2:  # some version of opencv spit out reversed result
            tmp = mask
            mask = lines_map
            lines_map = tmp
    mask = _original_postprocess_mask(mask)
    lines, scores = detector.seg_rep(detector.input_size, lines_map)
    box_thresh = 0.6
    idx = np.where(scores[0] > box_thresh)
    lines, scores = lines[0][idx], scores[0][idx]
    mask = mask[: mask.shape[0] - dh, : mask.shape[1] - dw]
    mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
    if lines.size == 0:
        lines = []
    else:
        lines = lines.astype(np.float64)
        lines[..., 0] *= resize_ratio[0]
        lines[..., 1] *= resize_ratio[1]
        lines = lines.astype(np.int32)
    blk_list = group_output(blks, lines, im_w, im_h, mask)
    mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
    if keep_undetected_mask:
        mask_refined = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)
    return mask, mask_refined, blk_list


def _fake_torch_net(blks, mask, lines_map):
    """Stand-in for TextDetBase: fixed outputs regardless of the input."""
    recorded = []

    def net(img_in):
        recorded.append(img_in)
        return blks.clone(), mask.clone(), lines_map.clone()

    net.recorded = recorded
    return net


def _fake_opencv_net(blks, mask, lines_map):
    recorded = []

    def net(img_in):
        recorded.append(img_in)
        return blks.copy(), mask.copy(), lines_map.copy()

    net.recorded = recorded
    return net


def _bare_detector(net, backend, input_size=(256, 256)):
    """A TextDetector without weights: only what __call__ touches."""
    det = TextDetector.__new__(TextDetector)
    det.model = None
    det.net = net
    det.backend = backend
    det.input_size = input_size
    det.device = 'cpu'
    det.half = False
    det.conf_thresh = 0.4
    det.nms_thresh = 0.35
    det.seg_rep = SegDetectorRepresenter(thresh=0.3)
    det.channels_last = False
    return det


def _fake_outputs():
    # (1, 3, 8): xywh + objectness + 3 class scores, letterbox coordinates
    blks = torch.tensor(
        [
            [
                [128.0, 128.0, 100.0, 100.0, 1.0, 0.9, 0.0, 0.0],
                [60.0, 60.0, 40.0, 30.0, 1.0, 0.0, 0.8, 0.0],
                [200.0, 40.0, 30.0, 20.0, 1.0, 0.0, 0.0, 0.7],
            ]
        ],
        dtype=torch.float32,
    )
    # the DB head outputs the mask at the letterbox resolution ...
    mask = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    mask[0, 0, 60:160, 70:170] = 0.87
    # ... and the text-line map at a quarter of it
    lines_map = torch.zeros(1, 2, 64, 64, dtype=torch.float32)
    lines_map[0, 0, 16:40, 16:40] = 0.91
    return blks, mask, lines_map


def _blk_signature(blk_list):
    sig = []
    for blk in blk_list:
        lines = blk.lines_array().tolist() if len(blk.lines) else []
        sig.append((blk.xyxy, blk.language, blk.vertical, lines))
    return sig


@pytest.mark.parametrize('shape', [(75, 100), (400, 300), (150, 200)])
@pytest.mark.parametrize('input_size', [(256, 256), (640, 640)])
def test_letterbox_input_decomposes_original_preprocess(shape, input_size):
    rng = np.random.default_rng(sum(shape) + input_size[0])
    img = rng.integers(0, 256, size=shape + (3,), dtype=np.uint8)

    img_u8, ratio, dw, dh = letterbox_input(img, input_size)
    new, new_ratio, new_dw, new_dh = preprocess_img(img, input_size=input_size)

    old, old_ratio, old_dw, old_dh = _original_preprocess_img(img, input_size=input_size)

    assert img_u8.dtype == np.uint8
    assert img_u8.shape == (1, 3, input_size[1], input_size[0])
    assert img_u8.flags['C_CONTIGUOUS']
    assert (ratio, dw, dh) == (old_ratio, old_dw, old_dh)
    assert (new_ratio, new_dw, new_dh) == (old_ratio, old_dw, old_dh)
    assert torch.equal(new, old)
    assert np.array_equal(new.numpy(), old.numpy())
    # same result whether the caller runs the two steps separately
    assert torch.equal(normalize_input(img_u8), old)


def test_normalize_input_accepts_host_tensor_and_half():
    rng = np.random.default_rng(7)
    img_u8 = rng.integers(0, 256, size=(1, 3, 256, 256), dtype=np.uint8)
    expected = torch.from_numpy(img_u8.astype(np.float32) / 255)

    assert torch.equal(normalize_input(img_u8), expected)
    assert torch.equal(normalize_input(torch.from_numpy(img_u8)), expected)
    assert torch.equal(normalize_input(img_u8, half=True), expected.half())


@pytest.mark.parametrize('shape', [(1, 1, 64, 80), (64, 80)])
def test_mask_to_uint8_matches_original_arithmetic(shape):
    rng = np.random.default_rng(3)
    values = rng.random(shape).astype(np.float32)
    values.flat[0] = 0.0
    values.flat[-1] = 1.0
    mask = torch.from_numpy(values.copy())

    old = _original_postprocess_mask(mask.clone())
    new = mask_to_uint8(mask)

    assert new.dtype == torch.uint8
    assert tuple(new.shape) == shape[-2:]
    assert np.array_equal(new.numpy(), old)
    assert np.array_equal(postprocess_mask(mask.clone()), old)


def test_postprocess_mask_tensor_threshold_path():
    rng = np.random.default_rng(11)
    mask = torch.from_numpy(rng.random((1, 1, 32, 32)).astype(np.float32))
    for thresh in (None, 0.3, 0.5):
        old = _original_postprocess_mask(mask.clone(), thresh=thresh)
        new = postprocess_mask(mask.clone(), thresh=thresh)
        assert np.array_equal(new, old)


def test_postprocess_mask_numpy_path():
    rng = np.random.default_rng(13)
    arr = rng.integers(0, 256, size=(1, 32, 32), dtype=np.uint8)
    for thresh in (None, 100):
        old = _original_postprocess_mask(arr.copy(), thresh=thresh)
        new = postprocess_mask(arr.copy(), thresh=thresh)
        assert np.array_equal(new, old)


def test_preprocess_img_to_tensor_false_keeps_rgb_letterbox():
    """The opencv-dnn path still converts BGR->RGB and returns a uint8 image."""
    rng = np.random.default_rng(23)
    img = rng.integers(0, 256, size=(120, 180, 3), dtype=np.uint8)
    new, new_ratio, new_dw, new_dh = preprocess_img(img, input_size=(256, 256), to_tensor=False)
    old, old_ratio, old_dw, old_dh = _original_preprocess_img(img, input_size=(256, 256), to_tensor=False)
    assert np.array_equal(new, old)
    assert (new_ratio, new_dw, new_dh) == (old_ratio, old_dw, old_dh)


def test_postprocess_yolo_matches_original():
    blks, _, _ = _fake_outputs()
    new = postprocess_yolo(blks.clone(), 0.4, 0.35, (0.78, 0.78))
    old = _original_postprocess_yolo(blks.clone(), 0.4, 0.35, (0.78, 0.78))
    for new_part, old_part in zip(new, old):
        assert np.array_equal(new_part, old_part)


def test_forward_returns_nms_detections_and_uint8_mask():
    blks, mask, lines_map = _fake_outputs()
    seen = {}

    def net(img_in):
        seen['input'] = img_in
        return blks.clone(), mask.clone(), lines_map.clone()

    det = _bare_detector(net, 'torch', input_size=(64, 64))
    img_in = torch.zeros(1, 3, 64, 64)
    det_out, mask_u8, lines_map_out = det.forward(img_in)

    assert det_out.dtype == torch.float32
    assert tuple(mask_u8.shape) == (256, 256)
    assert mask_u8.dtype == torch.uint8
    assert tuple(lines_map_out.shape) == (1, 1, 64, 64)
    # NMS keeps all three boxes and returns them in xyxy, conf, cls form
    assert tuple(det_out.shape) == (3, 6)
    assert seen['input'] is img_in


def test_forward_channels_last_is_applied():
    blks, mask, lines_map = _fake_outputs()
    seen = {}

    def net(img_in):
        seen['input'] = img_in
        return blks.clone(), mask.clone(), lines_map.clone()

    det = _bare_detector(net, 'torch', input_size=(64, 64))
    det.channels_last = True
    det.forward(torch.zeros(1, 3, 64, 64))
    assert seen['input'].is_contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize('keep_undetected_mask', [False, True])
def test_torch_call_matches_original_implementation(keep_undetected_mask):
    rng = np.random.default_rng(17)
    img = rng.integers(0, 256, size=(150, 200, 3), dtype=np.uint8)
    blks, mask, lines_map = _fake_outputs()

    new_det = _bare_detector(_fake_torch_net(blks, mask, lines_map), 'torch')
    old_det = _bare_detector(_fake_torch_net(blks, mask, lines_map), 'torch')

    new_mask, new_refined, new_blk_list = new_det(img, keep_undetected_mask=keep_undetected_mask)
    old_mask, old_refined, old_blk_list = _original_call(old_det, img, keep_undetected_mask=keep_undetected_mask)

    assert np.array_equal(new_mask, old_mask)
    assert np.array_equal(new_refined, old_refined)
    assert _blk_signature(new_blk_list) == _blk_signature(old_blk_list)
    # the network saw the same normalised input tensor
    assert torch.equal(new_det.net.recorded[0], old_det.net.recorded[0])


def test_opencv_call_matches_original_implementation():
    rng = np.random.default_rng(19)
    img = rng.integers(0, 256, size=(150, 200, 3), dtype=np.uint8)
    blks_t, mask_t, lines_t = _fake_outputs()
    blks = blks_t.numpy()
    mask = mask_t.numpy()
    lines_map = lines_t.numpy()

    new_det = _bare_detector(_fake_opencv_net(blks, mask, lines_map), 'opencv')
    old_det = _bare_detector(_fake_opencv_net(blks, mask, lines_map), 'opencv')

    new_mask, new_refined, new_blk_list = new_det(img, keep_undetected_mask=True)
    old_mask, old_refined, old_blk_list = _original_call(old_det, img, keep_undetected_mask=True)

    assert np.array_equal(new_mask, old_mask)
    assert np.array_equal(new_refined, old_refined)
    assert _blk_signature(new_blk_list) == _blk_signature(old_blk_list)
    assert np.array_equal(new_det.net.recorded[0], old_det.net.recorded[0])
