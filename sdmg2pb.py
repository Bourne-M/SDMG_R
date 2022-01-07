import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import cv2
import onnx
import os
import tensorflow as tf
import torch.jit
from onnx_tf.backend import prepare
import time
import random
from infer import read_data, data_prepare

import torch
from SDMG_Model import SDMG_R


def torch2pb():
    model = SDMG_R()
    to_use_device = torch.device('cpu')
    state_dict = torch.load('/path/to/your/workspace/test/best.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.to(to_use_device)
    model.eval()

    onnx_file_path = "./test_sdmg.onnx"
    relations = torch.randn(1, 10, 10, 5)
    texts = torch.randn(1, 10, 10)
    img = torch.randn(1, 3, 1024, 1024)
    boxes = torch.randn(1, 10, 4)

    relations = relations.to(to_use_device)
    texts = texts.to(to_use_device)
    dynamic_axes = {
        'img': {0: 'batch_size', 2: 'height', 3: "width"},
        'boxes': {0: 'batch_size', 1: "nums"},
        'relations': {0: 'batch_size', 1: 'nums', 2: 'nums'},
        'texts': {0: 'batch_size', 1: "nums", 2: "seq"},
        'seg_map': {0: 'batch_size', 1: "pro"},
        'seg_map2': {0: 'batch_size', 1: "pro"}
    }

    torch.onnx.export(model,  # model being run
                      (img, relations, texts, boxes),  # model input (or a tuple for multiple inputs)
                      onnx_file_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['img', 'relations', 'texts', 'boxes'],  # the model's input names
                      output_names=["seg_map", 'seg_map2'],  # the model's output names
                      dynamic_axes=dynamic_axes,
                      # verbose=True,
                      )

    print(f'onnx save to {onnx_file_path}')
    onnx_model = onnx.load(onnx_file_path)  # load onnx model
    print(onnx_model.graph.input)
    print(onnx_model.graph.output)
    tf_exp = prepare(onnx_model)
    tf_exp.export_graph('./pb/')
    print('pb model export successfully')


def test():
    model = tf.saved_model.load('./pb/')

    inference = model.signatures['serving_default']
    DATA_DIR = '/path/to/your/workspace/dataset/ceshi/image/'  # idcard
    save_dir = '/path/to/your/workspace/test/idcard-2/'
    os.makedirs(save_dir, exist_ok=True)

    data = read_data()
    total = 0
    for item in data:
        file_name = item['file_name']
        img_path = os.path.join(DATA_DIR, file_name)
        save_path = os.path.join(save_dir, file_name)
        img = cv2.imread(img_path)
        debug_img = img.copy()
        boxes1 = item['box']
        random.shuffle(boxes1)
        data_prepared = data_prepare(img, boxes1)
        img, relations, texts, boxes, tag = data_prepared

        img = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
        relations = tf.convert_to_tensor(np.expand_dims(relations, axis=0), dtype=tf.float32)
        texts = tf.convert_to_tensor(np.expand_dims(texts, axis=0), dtype=tf.float32)
        boxes = tf.convert_to_tensor(np.expand_dims(boxes, axis=0), dtype=tf.float32)
        stime = time.time()
        outputs = inference(img=img, relations=relations, texts=texts, boxes=boxes)
        cost = time.time() - stime
        total += cost
        s = outputs['output_0'].numpy().argmax(1)
        for idx, _b in enumerate(boxes1):
            pos = (_b['box'][0], _b['box'][1])
            str_cls = str(s[idx])
            cv2.putText(debug_img, str_cls, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(save_path, debug_img)

    print(total/len(data))


if __name__ == '__main__':
    torch2pb()
    test()
