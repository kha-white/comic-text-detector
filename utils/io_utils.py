import glob
import json
import os.path as osp
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def find_all_imgs(img_dir, abs_path=False):
    imglist = list()
    for filep in glob.glob(osp.join(img_dir, "*")):
        filename = osp.basename(filep)
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(filep)
        else:
            imglist.append(filename)
    return imglist

imread = lambda imgpath, read_type=cv2.IMREAD_COLOR: cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
# def imread(imgpath, read_type=cv2.IMREAD_COLOR):
#     img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
#     return img

def imwrite(img_path, img, ext='.png'):
    suffix = Path(img_path).suffix
    if suffix != '':
        img_path = img_path.replace(suffix, ext)
    else:
        img_path += ext
    cv2.imencode(ext, img)[1].tofile(img_path)