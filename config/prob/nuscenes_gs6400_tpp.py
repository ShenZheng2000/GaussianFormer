_base_ = [
    'nuscenes_gs6400.py',
]

model = dict(
    warp_type='tpp',   # ← add this line
    warp_input_shape=(864, 1600),
    warp_output_shape=(864, 1600), # NOTE: both are divisble by 32.
    vp_json = "/home/shenzhen/3D_Projects/neurvps/logs/output/vps.json",
    
    # debug_mode=True,
)