import os
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.models import builder
from mmdet3d.registry import MODELS as mmdet3dMODELS
import torch

from ...warp_utils.warping_layers import warp, apply_unwarp
from torchvision.utils import save_image


@MODELS.register_module()
class ResNetSecondFPN(BaseModule):
    def __init__(
        self, 
        img_backbone_config, 
        neck_confifg,
        img_backbone_out_indices,
        pretrained_path=None,
        lora_config=None,
        debug_mode=False,
    ):

        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone_config)
        self.img_neck = mmdet3dMODELS.build(neck_confifg)
        self.img_backbone_out_indices = img_backbone_out_indices
        self.debug_mode = debug_mode

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            print(self.load_state_dict(ckpt, strict=False))
            print("ResNetSecondFPN Weight Loaded Successfully.")

        # Apply peft LoRA to backbone (after loading pretrained weights)
        if lora_config is not None:
            from peft import get_peft_model, LoraConfig
            peft_config = LoraConfig(**lora_config)
            self.img_backbone = get_peft_model(self.img_backbone, peft_config)
            print(f"[LoRA] Applied to backbone: {peft_config}")

    def forward(self, imgs, warp_grid=None):

        # ★ WARP HERE (mirrors bev_segmentor.extract_img_feat)
        if warp_grid is not None:

            if self.debug_mode:
                os.makedirs('debug/tpp_lifter', exist_ok=True)
                print(f"[LIFTER WARP INPUT] imgs: {imgs.shape}")
                save_image(imgs[:6], 'debug/tpp_lifter/lifter_imgs_before_warp.png',
                           nrow=3, normalize=True)

            imgs = warp(warp_grid, imgs)

            if self.debug_mode:
                print(f"[LIFTER WARP OUTPUT] imgs: {imgs.shape}")
                save_image(imgs[:6], 'debug/tpp_lifter/lifter_imgs_after_warp.png',
                           nrow=3, normalize=True)

        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        secondfpn_out = self.img_neck(img_feats)[0]

        # ★ UNWARP HERE
        if warp_grid is not None:
            BN, C, H, W = secondfpn_out.shape

            if self.debug_mode:
                feat_before = secondfpn_out.mean(dim=1, keepdim=True)
                save_image(feat_before[:6],
                           f'debug/tpp_lifter/lifter_feat_before_unwarp_{H}x{W}.png',
                           nrow=3, normalize=True)
                print(f"[LIFTER UNWARP INPUT] secondfpn_out: {secondfpn_out.shape}")

            secondfpn_out = apply_unwarp(warp_grid, secondfpn_out, separable=True)

            if self.debug_mode:
                feat_after = secondfpn_out.mean(dim=1, keepdim=True)
                save_image(feat_after[:6],
                           f'debug/tpp_lifter/lifter_feat_after_unwarp_{H}x{W}.png',
                           nrow=3, normalize=True)
                print(f"[LIFTER UNWARP OUTPUT] secondfpn_out: {secondfpn_out.shape}")

        return secondfpn_out
