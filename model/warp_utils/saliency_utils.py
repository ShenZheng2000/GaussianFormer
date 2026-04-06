import json
import os


def load_vp_json(vp_json_path, top_crop=36):
    with open(vp_json_path) as f:
        raw = json.load(f)
    lookup = {}
    for cam, entries in raw.items():
        for rel_path, (u, v) in entries.items():
            lookup[rel_path] = [u, v - top_crop]
    return lookup


def get_vp(vp_lookup, full_path):
    key = next(k for k in vp_lookup if k in full_path)
    return vp_lookup[key]


def save_imgs_with_vp(imgs, vpts, img_filenames, save_path, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        idx = order[i]
        img_np = imgs[idx].cpu().float()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = TF.to_pil_image(img_np)
        u, v = vpts[idx].cpu().tolist()
        ax.imshow(img_np)
        ax.scatter(u, v, c='red', s=100, zorder=5)
        ax.set_title(img_filenames[idx].split('samples/')[-1][:30])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()