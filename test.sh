# NOTE: single-gpu for eval_fps.py, ALL gpus for eval.py!

 # v2 results (#6400) (Pretrained full-full)
python eval_fps.py --py-config config/prob/nuscenes_gs6400.py \
     --work-dir out/nuscenes_gs6400/ \
     --resume-from out/nuscenes_gs6400/state_dict.pth

# # v2 results (#6400) (Pretrained full-full, +TPP)
CUDA_VISIBLE_DEVICES=1 python eval_fps.py --py-config config/prob/nuscenes_gs6400_tpp.py \
     --work-dir out/nuscenes_gs6400_tpp/ \
     --resume-from out/nuscenes_gs6400/state_dict.pth

# v2 results (#6400, Pretrained full-half)
# CUDA_VISIBLE_DEVICES=2 python eval.py --py-config config/prob/nuscenes_gs6400_halfres.py \
#      --work-dir out/nuscenes_gs6400_halfres/ \
#      --resume-from out/nuscenes_gs6400/state_dict.pth

# v2 results ($6400, Pretrained full-half, +TPP)
CUDA_VISIBLE_DEVICES=3 python eval_fps.py --py-config config/prob/nuscenes_gs6400_tpp_halfres.py \
     --work-dir out/nuscenes_gs6400_tpp_halfres \
     --resume-from out/nuscenes_gs6400/state_dict.pth

# v2 results (#6400, Pretrained full-75)
# CUDA_VISIBLE_DEVICES=4 python eval.py --py-config config/prob/nuscenes_gs6400_75_res.py \
#      --work-dir out/nuscenes_gs6400_75_res/ \
#      --resume-from out/nuscenes_gs6400/state_dict.pth

# # v2 results ($6400, Pretrained full-75, +TPP)
CUDA_VISIBLE_DEVICES=5 python eval_fps.py --py-config config/prob/nuscenes_gs6400_tpp_75_res.py \
     --work-dir out/nuscenes_gs6400_tpp_75_res \
     --resume-from out/nuscenes_gs6400/state_dict.pth