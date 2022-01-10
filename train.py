import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from tqdm import tqdm
from SDMG_Model import SDMG_R
from SDMG_Loss import SDMGLoss
from utils import save_pth, compute_f1_score, Log
from SDMG_Dataset import SMGDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F


def evalu(model, val_loader, log, device):
    model.eval()
    node_gts = []
    node_preds = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            for k, val in data.items():
                if val is not None and isinstance(val, torch.Tensor):
                    data[k] = val.to(device)
            img, rel, txt, bbox, tag, lab = data['img'], data['relations'], data['texts'], data['gt_bboxes'], data['tag'], data['gt_labels']
            for batch_idx, _tag in enumerate(tag):
                node_gts.append(lab[batch_idx, :, 0])
            pred_nd, pred_eg = model.forward( img,rel, txt,bbox)
            pred_nd = F.softmax(pred_nd, -1)
            node_preds.append(pred_nd)
    node_preds = torch.cat(node_preds)
    node_gts = torch.cat(node_gts).int()
    node_f1s = compute_f1_score(node_preds, node_gts, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25,100])
    log.log_str(f'val:{node_f1s.mean()}')
    return node_f1s.mean()


def train(_model, _loss_func, _train_loader, _val_loader, _optim, _scheduler, _log, _device, _max_epoch, eval_iterval, model_save_dir):
    best_mode = {'f1': 0, 'best_epoch': 0}
    _log.log_str(f'start training ....')
    f1 = evalu(_model, _val_loader, _log, _device)
    for epoch in range(_max_epoch):
        for i, data in enumerate(_train_loader):
            _model.train()
            for k, val in data.items():
                if val is not None and isinstance(val, torch.Tensor):
                    data[k] = val.to(_device)
            img, rel, txt, bbox, tag, lab = data['img'], data['relations'], data['texts'], data['gt_bboxes'], data['tag'], data['gt_labels']
            _optim.zero_grad()
            pred_nd, pred_eg = _model.forward(img, rel, txt,bbox)

            losslab = []
            for batch_idx, _tag in enumerate(tag):
                tmplab = lab[batch_idx, :, :]
                losslab.append(tmplab)

            loss = _loss_func(pred_nd, pred_eg, losslab)
            loss['loss_all'].backward()
            _optim.step()
            _log.info(epoch, i, _scheduler.get_last_lr(), loss)
        if (epoch + 1) % eval_iterval == 0:
            f1 = evalu(_model, _val_loader, _log, _device)
            save_pth(model_save_dir + 'latest.pth', _model)
            if f1 > best_mode['f1']:
                tmp = {'f1': f1, 'best_epoch': epoch}
                best_mode.update(tmp)
                save_pth((model_save_dir + 'best.pth'), _model)
                _log.log_str(f'save pth to {model_save_dir + "best.pth"}')
        _log.clear()

        if epoch % 100 == 0:
            _scheduler.step()


if __name__ == '__main__':
    max_epoch = 90
    eval_iterval = 1
    print_step_iterval = 20
    to_use_device = torch.device('cuda')
    model_save_dir = '/path/to/your/workspace/test/'

    # train_label = '/path/to/your/workspace/dataset/wildreceipt/train.txt'
    # val_label = '/path/to/your/workspace/dataset/wildreceipt/test.txt'
    # dic_path = '/path/to/your/workspace/dataset/wildreceipt/dict.txt'
    # pic_root = '/path/to/your/workspace/dataset/wildreceipt'

    train_label = '/path/to/your/workspace/dataset/ceshi/train_new.txt'
    val_label = '/path/to/your/workspace/dataset/ceshi/test_new.txt'
    dic_path = '/path/to/your/workspace/dataset/ceshi/dict.txt'
    pic_root = '/path/to/your/workspace/dataset/ceshi/image/'

    train_dataset = SMGDataset(train_label, dic_path, pic_root)
    val_dataset = SMGDataset(val_label, dic_path, pic_root)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

    log = Log(max_epoch, len(train_loader), print_step_iterval)

    model = SDMG_R()
    model = model.to(to_use_device)

    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [20, 50], 0.1)
    loss_func = SDMGLoss()

    train(model, loss_func, train_loader, val_loader, optim, scheduler, log, to_use_device, max_epoch, eval_iterval, model_save_dir)
