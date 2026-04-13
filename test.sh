# NOTE: single-gpu for eval_fps.py, ALL gpus for eval.py!

eval_model() {
    local name=$1
    python eval.py \
        --py-config config/prob/${name}.py \
        --work-dir out/${name} \
        --resume-from out/nuscenes_gs6400/state_dict.pth
}

# Example usage: 
# eval_model nuscenes_gs6400_halfres
# eval_model nuscenes_gs6400_tpp_halfres
# eval_model nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn