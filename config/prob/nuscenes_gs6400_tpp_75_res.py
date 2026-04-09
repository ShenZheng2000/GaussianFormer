_base_ = [
    'nuscenes_gs6400_tpp.py',
]

model = dict(
    warp_output_shape=(640, 1184), # NOTE: both are divisble by 32. 
    lifter=dict(
       initializer_img_downsample=(640, 1184) # TODO: closest to warp output shape. 
    )
)