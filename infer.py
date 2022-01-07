import os.path

import torch
import json
import cv2
import numpy as np
from torch.nn import functional as F
from SDMG_Dataset import normalize, impad_to_multiple, sort_vertex8, _load_data
from SDMG_Model import SDMG_R
import random

model = SDMG_R()
to_use_device = torch.device('cuda')
state_dict = torch.load('/path/to/your/workspace/test/best.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=True)
model = model.to(to_use_device)
model.eval()

_char_path = '/path/to/your/workspace/dataset/ceshi/dict.txt'

char_dict = {
    '': 0,
    **{
        line.rstrip('\r\n'): ind
        for ind, line in enumerate(_load_data(_char_path), 1)
    }}


def pad_text_indices(text_inds, normal_seq_len=300):
    """Pad text index to same length."""
    real_seq_len = max([len(text_ind) for text_ind in text_inds])
    padded_text_inds = -np.ones((len(text_inds), 30), np.int32)
    for idx, text_ind in enumerate(text_inds):
        padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
    return padded_text_inds, real_seq_len


def compute_relation(boxes):
    """Compute relation between every two boxes."""
    # Get minimal axis-aligned bounding boxes for each of the boxes
    # yapf: disable
    bboxes = np.concatenate(
        [boxes[:, 0::2].min(axis=1, keepdims=True),
         boxes[:, 1::2].min(axis=1, keepdims=True),
         boxes[:, 0::2].max(axis=1, keepdims=True),
         boxes[:, 1::2].max(axis=1, keepdims=True)],
        axis=1).astype(np.float32)
    # yapf: enable
    x1, y1 = bboxes[:, 0:1], bboxes[:, 1:2]
    x2, y2 = bboxes[:, 2:3], bboxes[:, 3:4]
    w, h = np.maximum(x2 - x1 + 1, 1), np.maximum(y2 - y1 + 1, 1)
    dx = (x1.T - x1) / 10
    dy = (y1.T - y1) / 10
    xhh, xwh = h.T / h, w.T / h
    whs = w / h + np.zeros_like(xhh)
    relation = np.stack([dx, dy, whs, xhh, xwh], -1).astype(np.float32)
    return relation, bboxes


def read_data():
    txt_path = '/path/to/your/workspace/dataset/ceshi/test_new.txt'
    data_list = _load_data(txt_path)
    data_src = []
    for line_data in data_list:
        line_json = json.loads(line_data)
        item_dict = {}

        item_dict['file_name'] = line_json['file_name']
        item_dict['box'] = []
        for ann in line_json['annotations']:
            tmp_dic = {}
            tmp_dic['box'] = ann['box']
            tmp_dic['text'] = ann['text']
            item_dict['box'].append(tmp_dic)

        data_src.append(item_dict)
    return data_src


def data_prepare(_img, _boxes):
    standard_scale = [1024, 512]
    h, w, _ = _img.shape

    scale_factor = min(max(standard_scale) / max(h, w), min(standard_scale) / min(h, w))
    rh = int(h * float(scale_factor) + 0.5)
    rw = int(w * float(scale_factor) + 0.5)

    img = cv2.resize(_img, (rw, rh), interpolation=cv2.INTER_LINEAR)

    w_scale = img.shape[1] / _img.shape[1]
    h_scale = img.shape[0] / _img.shape[0]
    scale_a = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    n_img = normalize(img)
    p_img = impad_to_multiple(n_img)
    norm_pic = np.zeros((1024, 1024, 3), dtype=np.float32)
    norm_pic[:p_img.shape[0], :p_img.shape[1], :] = p_img

    boxes, texts, text_inds, labels, edges = [], [], [], [], []
    for _box in _boxes:
        box = _box['box']
        sorted_box = sort_vertex8(box[:8])
        boxes.append(sorted_box)
        text = _box['text']
        texts.append(text)
        text_ind = []
        for c in text:
            if c in char_dict:
                text_ind.append(char_dict[c])
            else:
                text_ind.append(3807)
        text_inds.append(text_ind)

    boxes = np.array(boxes, np.int32)
    relations, bboxes = compute_relation(boxes)
    padded_text_inds, seq_len = pad_text_indices(text_inds)

    new_bboxes = bboxes * scale_a
    new_bboxes[:, 0::2] = np.clip(new_bboxes[:, 0::2], 0, img.shape[1])
    new_bboxes[:, 1::2] = np.clip(new_bboxes[:, 1::2], 0, img.shape[0])

    # resize relations
    factor = np.array([w_scale, h_scale, w_scale / h_scale, 1, w_scale / h_scale]).astype(np.float32)
    relations = relations * factor[None, None]
    r_img = norm_pic.transpose(2, 0, 1)
    r_relation = relations
    r_text = padded_text_inds
    r_box = new_bboxes
    tag = np.array([len(new_bboxes), seq_len], dtype=np.int32)

    return r_img, r_relation, r_text, r_box, tag


def infer(_data):
    img, relations, texts, boxes, tag = _data
    with torch.no_grad():
        img = torch.tensor(img).unsqueeze(0)
        img = img.to(to_use_device)
        relations = torch.tensor(relations).unsqueeze(0).to(to_use_device)
        texts = torch.tensor(texts).unsqueeze(0).to(to_use_device)
        boxes = torch.tensor(boxes).unsqueeze(0).to(to_use_device)
        tag = torch.tensor(tag).unsqueeze(0).to(to_use_device)
        # pred_nd, pred_eg = model.forward(img, relations, texts, boxes, tag)
        pred_nd, pred_eg = model.forward( relations, texts)
        pred_nd = F.softmax(pred_nd, -1)
    return pred_nd.argmax(1)


if __name__ == '__main__':
    DATA_DIR = '/path/to/your/workspace/dataset/ceshi/image/'  # idcard
    save_dir = '/path/to/your/workspace/test/idcard/'
    os.makedirs(save_dir, exist_ok=True)

    data = read_data()
    for item in data:
        file_name = item['file_name']
        img_path = os.path.join(DATA_DIR, file_name)
        save_path = os.path.join(save_dir, file_name)
        img = cv2.imread(img_path)
        boxes = item['box']
        random.shuffle(boxes)
        data_prepared = data_prepare(img, boxes)
        cls = infer(data_prepared)
        cls = cls.detach().cpu().numpy().tolist()

        for idx, _b in enumerate(boxes):
            pos = (_b['box'][0], _b['box'][1])
            str_cls = str(cls[idx])
            cv2.putText(img, str_cls, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(save_path, img)
