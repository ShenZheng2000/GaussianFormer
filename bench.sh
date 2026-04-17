# NOTE: single-gpu for eval_fps.py

eval_model_pretrained() {
    local name=$1
    local script="eval.py"
    if [ "$2" = "--fps" ]; then
        script="eval_fps.py"
    fi
    echo "======== $name ========"
    python $script \
        --py-config config/prob/${name}.py \
        --work-dir out/${name} \
        --resume-from out/nuscenes_gs6400/state_dict.pth
    echo "========================"
}

# Example usage: 
# eval_model_pretrained   nuscenes_gs6400                                           --fps
# eval_model_pretrained   nuscenes_gs6400_tpp                                       --fps
# eval_model_pretrained   nuscenes_gs6400_halfres                                   --fps
# eval_model_pretrained   nuscenes_gs6400_quarterres                                --fps
# eval_model_pretrained   nuscenes_gs6400_tpp_halfres                               --fps
# eval_model_pretrained   nuscenes_gs6400_tpp_halfres_v2_unfreeze_lifter_nosyncbn   --fps
# eval_model_pretrained   nuscenes_gs6400_tpp_quarterres_v2                         --fps
# eval_model_pretrained   nuscenes_gs6400_75_res                                    --fps
# eval_model_pretrained   nuscenes_gs6400_tpp_75_res                                --fps