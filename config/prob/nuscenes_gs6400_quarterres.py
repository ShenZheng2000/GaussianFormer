_base_ = ['./nuscenes_gs6400.py']

# =========== data config ==============
input_shape = (416, 224)  # NOTE: both are divisble by 32.

# NOTE: need full {} here to avoid collapse. 
data_aug_conf = {
    "resize_lim": (0.26, 0.26), # NOTE: change this one, using 0.26 for both
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

val_dataset_config = dict(
    data_aug_conf=data_aug_conf
)

train_dataset_config = dict(
    data_aug_conf=data_aug_conf
)