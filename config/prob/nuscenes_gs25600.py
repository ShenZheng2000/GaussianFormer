# Inherits everything from gs6400 and overrides only what actually differs.
_base_ = ['./nuscenes_gs6400.py']
 
scale_range = [0.01, 1.8]  # gs6400 uses [0.01, 3.2]
 
model = dict(
    lifter=dict(
        num_anchor=19200,     # gs6400 uses 4000
        random_samples=6400,  # gs6400 uses 2400
    ),
    encoder=dict(
        deformable_model=dict(
            kps_generator=dict(
                scale_range=scale_range,
            ),
        ),
        refine_layer=dict(
            scale_range=scale_range,
        ),
    ),
)