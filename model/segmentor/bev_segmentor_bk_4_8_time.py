
from mmseg.models import SEGMENTORS
from mmseg.models import build_backbone

from .base_segmentor import CustomBaseSegmentor
import torch, time

from ..warp_utils.warping_layers import CuboidGlobalKDEGrid, warp, apply_unwarp
from ..warp_utils.saliency_utils import load_vp_json, get_vp, save_imgs_with_vp
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
        warp_type=None,
        warp_input_shape=(864, 1600),
        warp_output_shape=(864, 1600),
        vp_json=None,
        debug_mode=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # reorder for better visualization
        self.order = [2, 0, 1, 4, 3, 5]
        self.warp_type = warp_type
        self.debug_mode = debug_mode

        if self.warp_type == 'tpp':
            self.grid_net = CuboidGlobalKDEGrid(
                input_shape=warp_input_shape,
                output_shape=warp_output_shape,
                separable=True
            ).cuda()
            self.grid_net.requires_grad_(False)  # not learnable, not in state_dict issue

        # NOTE: hardcode top_crop as 36 now! 
        self.vp_lookup = load_vp_json(vp_json, top_crop=36) if vp_json is not None else None


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
        if self.warp_type == 'tpp':
            
            if self.debug_mode:
                print(f"[WARP INPUT] imgs: {imgs.shape}")                   # (B*N, 3, 864, 1600)
                print(f"[DEBUG] metas keys: {kwargs['metas'].keys()}")

            img_filenames = [f for batch in kwargs['metas']['img_filename'] for f in batch]  # flat list of B*N paths'
            
            t0 = time.perf_counter()
            vpts = imgs.new_tensor([get_vp(self.vp_lookup, p) for p in img_filenames])  # (B*N, 2)
            print(f"[TIME] get_vp: {(time.perf_counter()-t0)*1000:.2f}ms")

            torch.cuda.synchronize()
            t0 = time.perf_counter()            
            grid = self.grid_net(imgs, vpts)                            # (B*N, 864, 1600, 2)
            torch.cuda.synchronize()
            print("-------------------------->>>>>>>>>>>>>>>>")
            print(f"[TIME] grid_net: {(time.perf_counter()-t0)*1000:.2f}ms")


            # save original images (6 cameras, 2 rows x 3 cols)
            if self.debug_mode:
                os.makedirs('debug/tpp', exist_ok=True)
                save_image(imgs[self.order], 'debug/tpp/imgs_before_warp.png', nrow=3, normalize=True)
                save_imgs_with_vp(imgs, vpts, img_filenames, 'debug/tpp/imgs_before_warp_with_vp.png', self.order)  # ← insert here


            torch.cuda.synchronize()
            t0 = time.perf_counter()
            imgs = warp(grid, imgs)
            torch.cuda.synchronize()
            print(f"[TIME] warp: {(time.perf_counter()-t0)*1000:.2f}ms")

            
            # save warped images (6 cameras, 2 rows x 3 cols)
            if self.debug_mode:
                print(f"[WARP OUTPUT] imgs: {imgs.shape}")                  # (B*N, 3, 864, 1600)
                save_image(imgs[self.order], 'debug/tpp/imgs_after_warp.png', nrow=3, normalize=True)

                # save warped-unwarped images (should be same as original)
                imgs_unwarped = apply_unwarp(grid, imgs, separable=True)
                save_image(imgs_unwarped[self.order], 'debug/tpp/imgs_after_unwarp.png', nrow=3, normalize=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img_feats_backbone = self.img_backbone(imgs)
        torch.cuda.synchronize()
        print(f"[TIME] img_backbone: {(time.perf_counter()-t0)*1000:.2f}ms")

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
                if self.debug_mode:
                    feat_before = img_feat.mean(dim=1, keepdim=True)  # (B*N, 1, H, W)
                    save_image(feat_before[self.order], f'debug/tpp/feat_before_unwarp_{H}x{W}.png', nrow=3, normalize=True)
                    print(f"[UNWARP INPUT] img_feat: {img_feat.shape}")  # (B*N, C, H, W)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                img_feat = apply_unwarp(grid, img_feat, separable=True)
                torch.cuda.synchronize()
                print(f"[TIME] unwarp {H}x{W}: {(time.perf_counter()-t0)*1000:.2f}ms")

                # save feature after unwarp
                if self.debug_mode:
                    feat_after = img_feat.mean(dim=1, keepdim=True)  # (B*N, 1, H, W)
                    save_image(feat_after[self.order], f'debug/tpp/feat_after_unwarp_{H}x{W}.png', nrow=3, normalize=True)
                    print(f"[UNWARP OUTPUT] img_feat: {img_feat.shape}")

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

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = self.lifter(**results)
        torch.cuda.synchronize()
        print(f"[TIME] lifter: {(time.perf_counter()-t0)*1000:.2f}ms")

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