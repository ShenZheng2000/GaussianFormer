# v2 results (#6400, REtrained full-full) 
# python train.py --py-config config/prob/nuscenes_gs6400.py \
#     --work-dir /ssd0/shenzhen/Methods/GaussianFormer/out/nuscenes_gs6400

# v2 results (#6400, REtrained half-half)
# python train.py --py-config config/prob/nuscenes_gs6400_halfres.py \
#     --work-dir /ssd0/shenzhen/Methods/GaussianFormer/out/nuscenes_gs6400_halfres

# v2 results (#6400, REtrained half-half, +TPP)
python train.py --py-config config/prob/nuscenes_gs6400_tpp_halfres.py \
    --work-dir /ssd0/shenzhen/Methods/GaussianFormer/out/nuscenes_gs6400_tpp_halfres