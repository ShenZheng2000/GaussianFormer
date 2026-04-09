# ==========================================================================
# TPP warping config with LIFTER branch warping + LoRA adapters.
#
# This extends the base TPP config (nuscenes_gs6400_tpp.py) which only
# applies warping to the image backbone in extract_img_feat(). This config
# ALSO applies warping inside the lifter's ResNetSecondFPN initializer.
#
# Adaptation strategy:
#   - Backbone (ResNet-101, ~44.5M params): LoRA adapters (r=16) — preserves
#     pretrained features while adapting to warped input distribution.
#   - Neck (SECONDFPN, ~5.1M params): fully unfrozen — small enough
#     for full fine-tuning, and peft doesn't support ConvTranspose2d.
#
# lifter_input_mode controls how the lifter processes images:
#   'none'       — pass raw images directly (default, base config)
#   'downsample' — naive bilinear resize (see nuscenes_gs6400_tpp_halfres.py)
#   'warp'       — VP-guided TPP warp/unwarp (this config)
# ==========================================================================

_base_ = [
    'nuscenes_gs6400_tpp.py',
]

model = dict(
    freeze_lifter=True,           # LoRA handles backbone adaptation
    unfreeze_lifter_neck=True,    # fully unfreeze SECONDFPN neck (~5.1M params)
    lifter=dict(
        lifter_input_mode='warp',  # use TPP warping instead of downsampling
        initializer=dict(
            lora_config=dict(
                r=16,
                lora_alpha=32,
                target_modules=["conv1", "conv2", "conv3"],  # ResNet Bottleneck convs
                lora_dropout=0.0,
            ),
            debug_mode=False,  # set True to save lifter warp debug images
        ),
    ),
)
