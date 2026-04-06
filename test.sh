# # # # v2 results (#6400) (Pretrained full-full)
# python eval.py --py-config config/prob/nuscenes_gs6400.py \
#      --work-dir out/nuscenes_gs6400/ \
#      --resume-from out/nuscenes_gs6400/state_dict.pth

# # v2 results (#6400, Pretrained full-half)
# python eval.py --py-config config/prob/nuscenes_gs6400_halfres.py \
#      --work-dir out/nuscenes_gs6400/ \
#      --resume-from out/nuscenes_gs6400/state_dict.pth


# v2 results ($6400, Pretrained full-full, +TPP)

CUDA_VISIBLE_DEVICES=0 python eval.py --py-config config/prob/nuscenes_gs6400_tpp.py \
     --work-dir out/nuscenes_gs6400_tpp/ \
     --resume-from out/nuscenes_gs6400/state_dict.pth