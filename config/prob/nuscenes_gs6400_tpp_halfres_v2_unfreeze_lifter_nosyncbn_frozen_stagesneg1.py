_base_ = ['./nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn.py']


model = dict(
    img_backbone=dict(
        frozen_stages=-1,
    ),
    lifter=dict(
        initializer=dict(
            img_backbone_config=dict(
                frozen_stages=-1,
            )
        )
    )
)