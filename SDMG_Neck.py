from torch import nn
from torchvision.ops import RoIAlign
import torch


def box_roi(box_list):
    rois_list = []
    for img_id, bboxes in enumerate(box_list):
        # if bboxes.size(0) > 0:
        #     img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        #     rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        # else:
        #     rois = bboxes.new_zeros((0, 5))

        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)

        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


class SdmgNeck(nn.Module):
    def __init__(self):
        super(SdmgNeck, self).__init__()
        self.roi = RoIAlign((7, 7), spatial_scale=1, sampling_ratio=0,aligned=True)
        self.pool = nn.MaxPool2d(7)

    def forward(self, x, gt_boxes):
        x = self.roi(x, box_roi(gt_boxes))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
