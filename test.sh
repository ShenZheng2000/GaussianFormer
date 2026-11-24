# NOTE: ALL gpus for eval.py!
# TODO: later, use checkpoints of /scratch.

# NOTE: distance thresholding (near/mid/far mIoU) is DISABLED by default.
# To add it back: append `--spatial` to the python eval.py command in eval_model below
# (e.g. `--resume-from ... --spatial`).

eval_model() {
    local name=$1
    local epoch=${2:-20}
    python eval.py \
        --py-config config/prob/${name}.py \
        --work-dir out/${name} \
        --resume-from out/${name}/epoch_${epoch}.pth
}


# Example usage (with overall and day/night breakdowns): 
eval_model   nuscenes_gs6400                                                20
eval_model   nuscenes_gs6400_halfres                                        18
eval_model   nuscenes_gs6400_quarterres                                     18      
eval_model   nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn        20 
eval_model   nuscenes_gs6400_tpp_quarterres_v2                              17


# skip these for now
# eval_model   nuscenes_gs6400_tpp_halfres                                    17  
# eval_model   nuscenes_gs6400_tpp_halfres_unfreeze_lifter_nosyncbn           20
# eval_model   nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn_frozen_stages0   20
# eval_model   nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn_finetune   17 