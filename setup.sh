#!/bin/bash
# GaussianFormer Setup Reference

# ── 1. ENVIRONMENT FIXES ────────────────────────────────────
# (authors' guide missing these)
# module load cuda-11.8
# pip install einops jaxtyping kornia
# pointops fix → https://github.com/huang-yh/GaussianFormer/issues/47#issuecomment-4180099966

# ── 2. WEIGHTS ──────────────────────────────────────────────
# wget https://cloud.tsinghua.edu.cn/f/159a3370b4e843ddaec5/?dl=1

# ── 3. DATASET ──────────────────────────────────────────────
# SurroundOcc: skip Baidu, use GDrive instead
# gdown --folder https://drive.google.com/drive/folders/1N76ZOtLwarcVSo89xE8tDo_Boa1fh-_C

# ── 4. SYMLINKS ─────────────────────────────────────────────
# ln -s /ssd0/shenzhen/Datasets/nuscenes          /home/shenzhen/3D_Projects/GaussianFormer/data
# ln -s /ssd0/shenzhen/Datasets/nuscenes_cam      /home/shenzhen/3D_Projects/GaussianFormer/data
# ln -s /ssd0/shenzhen/Datasets/surroundocc       /home/shenzhen/3D_Projects/GaussianFormer/data



# Testing results that of current no use also put here for clarity. 
# v2 results (#12800) (retest: 20.09, report: 20.08)
# python eval.py --py-config config/prob/nuscenes_gs12800.py \
#      --work-dir out/nuscenes_gs12800/ \
#      --resume-from out/nuscenes_gs12800/state_dict.pth

# # # v2 results (#25600)  (retest: 20.36, report: 20.33)
# python eval.py --py-config config/prob/nuscenes_gs25600.py \
#         --work-dir out/nuscenes_gs25600/ \
#         --resume-from out/nuscenes_gs25600/state_dict.pth

# Issue: https://github.com/huang-yh/GaussianFormer/issues/82
# Cannot reproduce result (retest: 17.57, report: 19.31)
# python eval.py --py-config config/nuscenes_gs25600_solid.py \
#      --work-dir out/nuscenes_gs25600_solid/ \
#      --resume-from out/nuscenes_gs25600_solid/state_dict.pth