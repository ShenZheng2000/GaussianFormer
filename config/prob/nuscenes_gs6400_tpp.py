_base_ = [
    'nuscenes_gs6400.py',
]

model = dict(
    warp_type='tpp',   # ← add this line
)