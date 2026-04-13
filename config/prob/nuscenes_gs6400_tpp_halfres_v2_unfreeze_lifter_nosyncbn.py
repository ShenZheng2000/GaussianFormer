_base_ = ['./nuscenes_gs6400_tpp_halfres_v2.py']

model = dict(
    freeze_lifter=False,
)

# NOTE: this must be False, otherwise deadlock during training! 
syncBN = False