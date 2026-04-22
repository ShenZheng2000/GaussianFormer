train_model() {
    local name=$1
    python train.py \
        --py-config config/prob/${name}.py \
        --work-dir /ssd0/shenzhen/Methods/GaussianFormer/out/${name}
}

# Example usage: 
# train_model nuscenes_gs6400_halfres
# train_model nuscenes_gs6400_quarterres
# train_model nuscenes_gs6400_tpp_halfres
# train_model nuscenes_gs6400_tpp_halfres_unfreeze_lifter_nosyncbn
# train_model nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn
# train_model nuscenes_gs6400_tpp_quarterres_v2

# bad naming (change when train done): nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn_unfreeze_stage0
# train_model nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn_frozen_stages0
# train_model nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn_frozen_stagesneg1