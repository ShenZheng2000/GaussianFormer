_base_ = [
    'nuscenes_gs6400_tpp.py',
]

model = dict(
    warp_output_shape=(640, 1184), # NOTE: both are divisble by 32. 
)