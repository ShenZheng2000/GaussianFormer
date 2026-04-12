_base_ = ['./nuscenes_gs6400_halfres.py']

model = dict(
    freeze_lifter=False,
)

# NOTE: this must be False, otherwise deadlock during training! 
syncBN = False