from torch import nn
from SDMG_Head import SDMGRHead
from SDMG_Neck import SdmgNeck
from SDMG_Backbone import UNet


class SDMG_R(nn.Module):
    def __init__(self):
        super(SDMG_R, self).__init__()
        # self.backbone = UNet(base_channels=16)
        # self.neck = SdmgNeck()
        self.head = SDMGRHead()

    def _prepare(self, relations, texts):
        rela = []
        txt = []
        for batch_idx, _tag in enumerate(texts):
            rela.append(relations[batch_idx, :, :, :])
            txt.append(texts[batch_idx, :, :])

        return rela, txt

    def forward(self, relations, texts):
        rela, txt = self._prepare(relations, texts)

        # x = self.backbone(img)
        # x = self.neck(x, bbox)
        # x = self.head(rela, txt, x)
        x = self.head(rela, txt, None)
        return x
