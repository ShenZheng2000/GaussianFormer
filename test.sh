# # # # v2 results (#6400) (Pretrained full-full)
# python eval.py --py-config config/prob/nuscenes_gs6400.py \
#      --work-dir out/nuscenes_gs6400/ \
#      --resume-from out/nuscenes_gs6400/state_dict.pth

# # v2 results (#6400, Pretrained full-half)
python eval.py --py-config config/prob/nuscenes_gs6400_halfres.py \
     --work-dir out/nuscenes_gs6400_halfres/ \
     --resume-from out/nuscenes_gs6400/state_dict.pth

# # v2 results (#6400, Pretrained full-75)
python eval.py --py-config config/prob/nuscenes_gs6400_75_res.py \
     --work-dir out/nuscenes_gs6400_75_res/ \
     --resume-from out/nuscenes_gs6400/state_dict.pth

# v2 results ($6400, Pretrained full-half, +TPP)
python eval.py --py-config config/prob/nuscenes_gs6400_tpp_halfres.py \
     --work-dir out/nuscenes_gs6400_tpp_halfres \
     --resume-from out/nuscenes_gs6400/state_dict.pth

# # v2 results ($6400, Pretrained full-75, +TPP)
python eval.py --py-config config/prob/nuscenes_gs6400_tpp_75_res.py \
     --work-dir out/nuscenes_gs6400_tpp_75_res \
     --resume-from out/nuscenes_gs6400/state_dict.pth