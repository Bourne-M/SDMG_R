import torch
import numpy as np
from collections import OrderedDict


def save_pth(_path, _model):
    model_state_dict = _model.state_dict()
    torch.save(model_state_dict, _path)


def compute_f1_score(preds, gts, ignores=[]):
    C = preds.size(1)
    classes = torch.LongTensor(sorted(set(range(C)) - set(ignores)))
    mask = (gts >= 0) & (gts < 20)

    hist = torch.bincount(gts[mask] * C + preds.argmax(1)[mask], minlength=C ** 2).view(C, C).float()
    diag = torch.diag(hist)
    recalls = diag / hist.sum(1).clamp(min=1)
    precisions = diag / hist.sum(0).clamp(min=1)
    f1 = 2 * recalls * precisions / (recalls + precisions).clamp(min=1e-8)
    return f1[classes].cpu().numpy()


class Log:
    def __init__(self, _epoch, _all_step, interval=10):
        self.count = -1
        self.interval = interval
        self.all_epoch = _epoch
        self.all_step = _all_step
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output_dic = OrderedDict()

    def info(self, _epoch, _step, _lr, _log_dic, record_type='step'):
        log_str = ''
        log_str += self.log_step(_log_dic)
        if self.count % self.interval == 0:
            prt_st = f'[{_epoch}/{self.all_epoch} ] [{_step}/{self.all_step}] '
            print(prt_st + log_str)
        elif self.count % self.all_step == 0:
            prt_st = f'[{_epoch}/{self.all_epoch} ] done'
            print(prt_st + log_str)

    def log_str(self, str):
        print(str)

    def log_step(self, vars):
        out_str = ''
        self.update(vars)
        if self.count % self.interval == 0:
            self.averge(self.interval)

        elif self.count % self.all_step == 0:
            self.averge(self.all_step)
            self.clear()

        for key, val in self.output_dic.items():
            out_str += f'{key}:{val:.3f} '
        return out_str

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, val in vars.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.val_history.setdefault(key, []).append(val)
            self.n_history.setdefault(key, []).append(count)
        self.count += 1

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.output_dic.clear()
        self.count = -1

    def averge(self, n=0):
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output_dic[key] = avg
