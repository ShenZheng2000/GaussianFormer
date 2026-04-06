
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone

from .base_segmentor import CustomBaseSegmentor
import torch, time

from ..warp_utils.warping_layers import CuboidGlobalKDEGrid, warp, apply_unwarp
from torchvision.utils import save_image
import os

@SEGMENTORS.register_module()
class BEVSegmentor(CustomBaseSegmentor):

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        # use_post_fusion=False,
        warp_type=None,  # None: no warping, 'tpp': use CuboidGlobalKDEGrid
        **kwargs,
    ):
        super().__init__(**kwargs)


        # reorder for better visualization
        self.order = [2, 0, 1, 4, 3, 5]

        # TODO: hardcode tpp with nuscene preprocessed image size for now!
        self.warp_type = warp_type

        if self.warp_type == 'tpp':
            self.grid_net = CuboidGlobalKDEGrid(
                input_shape=(864, 1600),
                output_shape=(864, 1600),
                separable=True
            ).cuda()
            self.grid_net.requires_grad_(False)  # not learnable, not in state_dict issue


        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        # self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)


        # ★ WARP HERE
        # TODO: hardcode dummy vp now, use real-dataset vp later!  
        if self.warp_type == 'tpp':
            
            print(f"[WARP INPUT] imgs: {imgs.shape}")                   # (B*N, 3, 864, 1600)
            vpts = imgs.new_tensor([[W / 2, H / 2]]).expand(B * N, -1)  # (B*N, 2)
            grid = self.grid_net(imgs, vpts)                            # (B*N, 864, 1600, 2)

            # save original images (6 cameras, 2 rows x 3 cols)
            os.makedirs('debug/tpp', exist_ok=True)
            save_image(imgs[self.order], 'debug/tpp/imgs_before_warp.png', nrow=3, normalize=True)

            imgs = warp(grid, imgs)
            print(f"[WARP OUTPUT] imgs: {imgs.shape}")                  # (B*N, 3, 864, 1600)

            # save warped images (6 cameras, 2 rows x 3 cols)
            save_image(imgs[self.order], 'debug/tpp/imgs_after_warp.png', nrow=3, normalize=True)

            # debug: unwarp the warped image back, should look close to original
            imgs_unwarped = apply_unwarp(grid, imgs, separable=True)
            save_image(imgs_unwarped[self.order], 'debug/tpp/imgs_after_unwarp.png', nrow=3, normalize=True)


        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()


            # ★ UNWARP here
            if self.warp_type == 'tpp':

                # save feature before unwarp (average over channel dim → grayscale)
                feat_before = img_feat.mean(dim=1, keepdim=True)  # (B*N, 1, H, W)
                save_image(feat_before[self.order], f'debug/tpp/feat_before_unwarp_{H}x{W}.png', nrow=3, normalize=True)
                
                print(f"[UNWARP INPUT] img_feat: {img_feat.shape}")  # (B*N, C, H, W)
                img_feat = apply_unwarp(grid, img_feat, separable=True)
                print(f"[UNWARP OUTPUT] img_feat: {img_feat.shape}")
                
                # save feature after unwarp
                feat_after = img_feat.mean(dim=1, keepdim=True)  # (B*N, 1, H, W)
                save_image(feat_after[self.order], f'debug/tpp/feat_after_unwarp_{H}x{W}.png', nrow=3, normalize=True)

                exit()


            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result
    
    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function.
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)

        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        outs = self.lifter(**results)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # results.update({"lifter_time": elapsed})

        results.update(outs)
        outs = self.encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results