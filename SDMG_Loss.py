import torch
from torch import nn


def accuracy(pred, tareget, topk=1, thresh=None):
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)

    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # shape(maxk,N)
    correct = pred_label.eq(tareget.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul(100.0 / pred.size(0)))
    return res[0] if return_single else res


class SDMGLoss(nn.Module):
    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=-100):
        super(SDMGLoss, self).__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=100)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=100)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def forward(self, node_preds, edge_preds, gts):
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].contiguous().view(-1))
        node_gts = torch.cat(node_gts).long()
        edge_gts = torch.cat(edge_gts).long()

        node_valids = torch.nonzero(node_gts != 100, as_tuple=False).view(-1)
        edge_valids = torch.nonzero(edge_gts != 100, as_tuple=False).view(-1)
        n_loss = self.node_weight * self.loss_node(node_preds, node_gts)
        e_loss = self.edge_weight * self.loss_edge(edge_preds, edge_gts)
        a_loss = n_loss + e_loss
        res = dict(
            loss_all=a_loss,
            loss_node=n_loss,
            loss_edge=e_loss,
            acc_node=accuracy(node_preds[node_valids], node_gts[node_valids]),
            acc_edge=accuracy(edge_preds[edge_valids], edge_gts[edge_valids])
        )
        return res
