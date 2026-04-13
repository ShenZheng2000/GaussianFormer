_base_ = [
    'nuscenes_gs6400_tpp.py',
]

model = dict(
    warp_output_shape=(448, 832),
    lifter=dict(
        lifter_input_mode='warp',
    ),
)