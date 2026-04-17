import time, argparse, os.path as osp, os
import torch
import warnings
warnings.filterwarnings("ignore")

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
import logging


NUM_SAMPLES = 500
WARMUP      = 50   # first N samples discarded


def main(args):
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file  = osp.join(args.work_dir, f'latency_{timestamp}.log')
    # logger    = MMLogger('latency', log_file=log_file)

    # NOTE: STOP saving log_file for cleaner output.
    # logger    = MMLogger('latency', log_file=None)
    logger = MMLogger.get_instance('latency', log_file=None, log_level='WARNING')
    # MMLogger._instance_dict['latency'] = logger
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model — single GPU only
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)

    # NOTE: fix long printout of init_weights() for cleaner output.
    logger.setLevel(logging.WARNING)
    my_model.init_weights()
    # logger.setLevel(logging.INFO)

    my_model = my_model.cuda()
    my_model.eval()
    os.environ['eval'] = 'true'

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    # load checkpoint
    resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        resume_from = args.resume_from

    if resume_from and osp.exists(resume_from):
        ckpt = torch.load(resume_from, map_location='cpu')
        my_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        logger.info(f'Loaded checkpoint from {resume_from}')
    elif cfg.get('load_from', None):
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        my_model.load_state_dict(state_dict, strict=False)
        logger.info(f'Loaded from cfg.load_from: {cfg.load_from}')

    # dataloader — val only, batch size 1, single GPU
    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        val_only=True)

    latencies = []
    mem_usages = []   # ← add here

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):

            if i_iter_val >= NUM_SAMPLES:
                break

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')

            # --- time the forward pass ---
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()   # ← add here
            t0 = time.perf_counter()

            result_dict = my_model(imgs=input_imgs, metas=data)

            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            mem_usages.append(torch.cuda.max_memory_allocated() / 1024 ** 3)   # GB
            # -----------------------------

            if i_iter_val % 10 == 0:
                print(f'[LATENCY] sample {i_iter_val}/{NUM_SAMPLES}')

    # discard warmup
    latencies = latencies[WARMUP:]
    avg_ms  = (sum(latencies) / len(latencies)) * 1000
    fps     = 1.0 / (sum(latencies) / len(latencies))

    mem_usages = mem_usages[WARMUP:]   # ← add here
    avg_mem  = sum(mem_usages) / len(mem_usages)
    peak_mem = max(mem_usages)    

    print(f'--- Latency Results (samples={len(latencies)}, warmup={WARMUP}) ---')
    print(f'Avg latency : {avg_ms:.2f} ms')
    print(f'FPS         : {fps:.2f}')

    print(f'Avg memory  : {avg_mem:.2f} GB')
    print(f'Peak memory : {peak_mem:.2f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latency measurement — single GPU, batch size 1')
    parser.add_argument('--py-config',    default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir',     type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from',  type=str, default='')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    # force single GPU
    assert torch.cuda.is_available(), 'CUDA required'
    main(args)