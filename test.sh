# NOTE: ALL gpus for eval.py!

eval_model() {
    local name=$1
    local epoch=${2:-20}
    python eval.py \
        --py-config config/prob/${name}.py \
        --work-dir out/${name} \
        --resume-from out/${name}/epoch_${epoch}.pth
}

# Example usage: 
# eval_model   nuscenes_gs6400_halfres                                        18
# eval_model   nuscenes_gs6400_quarterres                                     18
# eval_model   nuscenes_gs6400_tpp_halfres                                    17        
# eval_model   nuscenes_gs6400_tpp_halfres_unfreeze_lifter_nosyncbn           20
# eval_model   nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn        20 
# eval_model   nuscenes_gs6400_tpp_quarterres_v2                              17