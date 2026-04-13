from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.models import builder
from mmdet3d.registry import MODELS as mmdet3dMODELS
import torch
from ...warp_utils.warping_layers import warp, apply_unwarp
import torch.nn.functional as F
import os
from torchvision.utils import save_image

@MODELS.register_module()
class ResNetSecondFPN(BaseModule):
    def __init__(
        self, 
        img_backbone_config, 
        neck_confifg,
        img_backbone_out_indices,
        pretrained_path=None,
        debug_mode=False,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone_config)
        self.img_neck = mmdet3dMODELS.build(neck_confifg)
        self.img_backbone_out_indices = img_backbone_out_indices
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            print(self.load_state_dict(ckpt, strict=False))
            print("ResNetSecondFPN Weight Loaded Successfully.")

        # reorder for better visualization
        self.order = [2, 0, 1, 4, 3, 5]
        self.debug_mode = debug_mode

    def forward(self, imgs, warp_grid=None,):

        # ★ WARP HERE
        if warp_grid is not None:
            
            # save original images
            if self.debug_mode:
                os.makedirs('debug/tpp_lifter', exist_ok=True)
                save_image(imgs[self.order], 'debug/tpp_lifter/imgs_before_warp.png', nrow=3, normalize=True)

            imgs = warp(warp_grid, imgs)

            # save warped images
            if self.debug_mode:
                save_image(imgs[self.order], 'debug/tpp_lifter/imgs_after_warp.png', nrow=3, normalize=True)

                # save warped-unwarped images (should be same as original)
                imgs_unwarped = apply_unwarp(warp_grid, imgs, separable=True)
                save_image(imgs_unwarped[self.order], 'debug/tpp_lifter/imgs_after_unwarp.png', nrow=3, normalize=True)


        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        secondfpn_out = self.img_neck(img_feats)[0]


        # ★ UNWARP here
        if warp_grid is not None:

            # save feature before unwarp
            if self.debug_mode:
                BN, C, H, W = secondfpn_out.shape
                save_image(secondfpn_out.mean(1, keepdim=True)[self.order],
                           f'debug/tpp_lifter/feat_before_unwarp_{H}x{W}.png', nrow=3, normalize=True)

            secondfpn_out = apply_unwarp(warp_grid, secondfpn_out, separable=True)

            # save feature after unwarp
            if self.debug_mode:
                save_image(secondfpn_out.mean(1, keepdim=True)[self.order],
                           f'debug/tpp_lifter/feat_after_unwarp_{H}x{W}.png', nrow=3, normalize=True)

                exit()


        return secondfpn_out
