import json
import os
import random

import cv2
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


def sort_vertex8(points):
    """Sort vertex with 8 points [x1 y1 x2 y2 x3 y3 x4 y4]"""
    assert len(points) == 8
    vertices = _sort_vertex(np.array(points, dtype=np.float32).reshape(-1, 2))
    sorted_box = list(vertices.flatten())
    return sorted_box


def _sort_vertex(vertices):
    assert vertices.ndim == 2
    assert vertices.shape[-1] == 2
    N = vertices.shape[0]
    if N == 0:
        return vertices

    center = np.mean(vertices, axis=0)
    directions = vertices - center
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    sort_idx = np.argsort(angles)
    vertices = vertices[sort_idx]

    left_top = np.min(vertices, axis=0)
    dists = np.linalg.norm(left_top - vertices, axis=-1, ord=2)
    lefttop_idx = np.argmin(dists)
    indexes = (np.arange(N, dtype=np.int) + lefttop_idx) % N
    return vertices[indexes]


def normalize(img):
    mat = img.copy().astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    mean = np.float64(mean.reshape((1, -1)))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(mat, cv2.COLOR_BGR2RGB, mat)
    cv2.subtract(mat, mean, mat)
    cv2.multiply(mat, stdinv, mat)
    return mat


def impad_to_multiple(img, divisor=32, pad_val=0):
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor - img.shape[0]
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor - img.shape[1]
    dst = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, pad_val)
    return dst


def _load_data(_path, encoding='utf-8'):
    data_list = []
    with open(_path, 'r', encoding=encoding) as f:
        for line in f:
            data_list.append(line.rstrip('\n\r'))
    return data_list


class SMGDataset(Dataset):
    def __init__(self, _path, _char_path, _pic_root, scale=None, _norm=10, _max_content_len=25, _max_node_num=30,_ignore=100):
        super(SMGDataset, self).__init__()
        if scale is None:
            scale = [1024, 512]
        self.data_list = _load_data(_path)
        self.norm = _norm
        self.pic_root = _pic_root
        self.dict = {
            '': 0,
            **{
                line.rstrip('\r\n'): ind
                for ind, line in enumerate(_load_data(_char_path), 1)
            }}
        self.directed = False
        self.scale = scale
        self.max_content_len = _max_content_len
        self.max_node_num = _max_node_num
        self.ignore=_ignore

    def _str2_json(self, _str):
        res = {}
        line_json_obj = json.loads(_str)
        for key in ['file_name', 'height', 'width', 'annotations']:
            res[key] = line_json_obj[key]
        random.shuffle(res['annotations'])
        return res

    def __getitem__(self, index):
        result_dict = {}
        img_ann_info = self._str2_json(self.data_list[index])

        ann_info = self._parse_ann_info(img_ann_info['annotations'])

        img_path = os.path.join(self.pic_root, img_ann_info['file_name'])
        img_src = cv2.imread(img_path)
        result_dict['filename'] = img_ann_info['file_name']
        result_dict['gt_bboxes'] = ann_info['bboxes'].copy()
        result_dict['gt_labels'] = ann_info['labels'].copy()
        result_dict['relations'] = ann_info['relations']
        result_dict['texts'] = ann_info['texts']
        result_dict['tag'] = ann_info['tag']

        # resize and pad img with longside
        h, w, _ = img_src.shape
        scale_factor = min(max(self.scale) / max(h, w), min(self.scale) / min(h, w))
        rh = int(h * float(scale_factor) + 0.5)
        rw = int(w * float(scale_factor) + 0.5)
        img = cv2.resize(img_src, (rw, rh), interpolation=cv2.INTER_LINEAR)

        w_scale = img.shape[1] / img_src.shape[1]
        h_scale = img.shape[0] / img_src.shape[0]
        scale_a = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        n_img = normalize(img)
        p_img = impad_to_multiple(n_img)
        norm_pic = np.zeros((max(self.scale), max(self.scale), 3), dtype=np.float32)
        norm_pic[:p_img.shape[0], :p_img.shape[1], :] = p_img
        norm_pic = norm_pic.transpose((2, 0, 1))
        result_dict['img'] = norm_pic
        # resize gtbox
        new_bboxes = result_dict['gt_bboxes'] * scale_a
        new_bboxes[:, 0::2] = np.clip(new_bboxes[:, 0::2], 0, img.shape[1])
        new_bboxes[:, 1::2] = np.clip(new_bboxes[:, 1::2], 0, img.shape[0])
        result_dict['gt_bboxes'] = new_bboxes
        # resize relations
        factor = np.array([w_scale, h_scale, w_scale / h_scale, 1, w_scale / h_scale]).astype(np.float32)
        result_dict['relations'] = result_dict['relations'] * factor[None, None]
        return result_dict

    def _parse_ann_info(self, annotations):
        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['box']
            sorted_box = sort_vertex8(box[:8])
            boxes.append(sorted_box)
            text = ann['text']
            texts.append(ann['text'])
            text_ind = [self.dict[c] for c in text if c in self.dict]

            # text_ind = []
            # for c in text:
            #     if c in self.dict:
            #         text_ind.append(self.dict[c])
            #     else:
            #         text_ind.append(3807)

            text_inds.append(text_ind)
            labels.append(ann.get('label', 0))
            edges.append(ann.get('edge', 0))

        ann_infos = dict(
            boxes=boxes,
            texts=texts,
            text_inds=text_inds,
            edges=edges,
            labels=labels)
        return self.list_to_numpy(ann_infos)

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        real_seq_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = -np.ones((len(text_inds), self.max_content_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds, real_seq_len

    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""
        boxes, text_inds = ann_infos['boxes'], ann_infos['text_inds']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, self.ignore)
                labels = np.concatenate([labels, edges], -1)
        padded_text_inds, seq_len = self.pad_text_indices(text_inds)

        temp_bboxes = np.zeros([self.max_node_num, 4], dtype=np.float32)
        h, _ = bboxes.shape
        temp_bboxes[:h, :] = bboxes

        temp_relations = np.zeros([self.max_node_num, self.max_node_num, 5], dtype=np.float32)
        temp_relations[:h, :h, :] = relations

        temp_padded_text_inds = np.zeros([self.max_node_num, self.max_node_num], dtype=np.float32)
        temp_padded_text_inds[:h, :] = padded_text_inds

        temp_labels = np.ones([self.max_node_num, self.max_node_num + 1], dtype=np.int32) * self.ignore
        temp_labels[:h, :h + 1] = labels

        tag = np.array([h, seq_len])

        return dict(
            bboxes=temp_bboxes,
            relations=temp_relations,
            texts=temp_padded_text_inds,
            labels=temp_labels,
            tag=tag
        )

    def compute_relation(self, boxes):
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
        dx = (x1.T - x1) / self.norm
        dy = (y1.T - y1) / self.norm
        xhh, xwh = h.T / h, w.T / h
        whs = w / h + np.zeros_like(xhh)
        relation = np.stack([dx, dy, whs, xhh, xwh], -1).astype(np.float32)
        return relation, bboxes

    def __len__(self):
        return len(self.data_list)


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


def my_collate(batch):
    max_shape = [0, 0]
    for data in batch:
        for dim in range(2):
            max_shape[dim] = max(max_shape[dim], data['img'].shape[dim])
    padded_img = []
    relation = []
    texts = []
    gt_bboxes = []
    gt_label = []
    for data in batch:
        big_pic = np.zeros((3, max_shape[0], max_shape[1]))
        src_img = data['img'].transpose(2, 0, 1)
        h, w, _ = src_img.shape
        big_pic[:h, :w, :] = src_img
        padded_img.append(big_pic)
        relation.append(data['relations'])
        texts.append(data['texts'])
        gt_bboxes.append(data['gt_bboxes'])
        gt_label.append(data['gt_labels'])
    return_img = default_collate(padded_img)
    relation = default_collate(relation)
    texts = default_collate(texts)
    gt_bboxes = default_collate(gt_bboxes)
    gt_label = default_collate(gt_label)

    return return_img, relation, texts, gt_bboxes, gt_label


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    train_label = '/path/to/your/workspace/dataset/wildreceipt/train.txt'
    val_label = '/path/to/your/workspace/dataset/wildreceipt/test.txt'
    dic_path = '/path/to/your/workspace/dataset/wildreceipt/dict.txt'
    pic_root = '/path/to/your/workspace/dataset/wildreceipt'

    # train_label = '/path/to/your/workspace/dataset/ceshi/train_new.txt'
    # val_label = '/path/to/your/workspace/dataset/ceshi/test_new.txt'
    # dic_path = '/path/to/your/workspace/dataset/ceshi/dict.txt'
    # pic_root = '/path/to/your/workspace/dataset/ceshi/image/'

    train_dataset = SMGDataset(train_label, dic_path, pic_root)
    val_dataset = SMGDataset(val_label, dic_path, pic_root)
    train_loader = DataLoader(dataset=train_dataset, batch_size=3, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=6)

    for i, data in enumerate(tqdm(train_loader)):
        fn = data['filename']
        bboxes = data['gt_bboxes']
        labels = data['gt_labels']
        rel = data['relations']
        txt = data['texts']
        tag = data['tag']

        pass
