# ==========================================================================
# TPP warping in BOTH image branch and lifter branch — fully frozen.
#
# No LoRA, no unfreezing. Just applies warp/unwarp around both backbones
# using the pretrained weights as-is. Useful for evaluating the effect of
# warping alone without any training.
# ==========================================================================

_base_ = [
    'nuscenes_gs6400_tpp.py',
]

model = dict(
    freeze_lifter=True,  # everything stays frozen, no adaptation
    lifter=dict(
        lifter_input_mode='warp',  # warp in lifter branch (image branch already warped by base TPP config)
    ),
)
