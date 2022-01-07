from torch import nn
from SDMG_Head import SDMGRHead
from SDMG_Neck import SdmgNeck
from SDMG_Backbone import UNet


class SDMG_R(nn.Module):
    def __init__(self):
        super(SDMG_R, self).__init__()
        self.backbone = UNet(base_channels=16)
        self.neck = SdmgNeck()
        self.head = SDMGRHead()

    def _prepare(self, pic, relations, texts, gt_bboxes):
        batch_pic = pic
        rela = []
        txt = []
        bboxes = []
        for batch_idx, _tag in enumerate(texts):
            rela.append(relations[batch_idx, :, :, :])
            txt.append(texts[batch_idx, :, :])
            bboxes.append(gt_bboxes[batch_idx, :, :])

        return batch_pic, rela, txt, bboxes

    def forward(self, pic, relations, texts, gt_bboxes):
        img, rela, txt, bbox = self._prepare(pic, relations, texts, gt_bboxes)

        x = self.backbone(img)
        x = self.neck(x, bbox)
        x = self.head(rela, txt, x)
        return x
