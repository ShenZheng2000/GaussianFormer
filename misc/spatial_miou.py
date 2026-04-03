"""
Spatial mIoU Tracker: evaluates near/far mIoU at multiple thresholds
in a single eval pass. Drop-in utility for eval.py.

Usage in eval.py:
    tracker = SpatialMIoUTracker(near_thresholds, far_thresholds)
    # in loop:
    tracker.update(pred_occ, gt_occ, sampled_xyz, occ_mask)
    # after loop:
    tracker.report(logger)
"""

import numpy as np
import torch
import torch.distributed as dist

from misc.metric_util import MeanIoU


LABEL_STR = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]
EMPTY_LABEL = 17

# Default thresholds (meters from ego) — edit these to change the sweep
NEAR_THRESHOLDS = [10, 15, 20]   # mIoU computed for voxels within this distance
FAR_THRESHOLDS = [30, 35, 40]    # mIoU computed for voxels beyond this distance


class SpatialMIoUTracker:
    """Tracks near/far mIoU across multiple thresholds and distance bins."""

    def __init__(self, near_thresholds=None, far_thresholds=None,
                 bin_size=5, max_range=50):
        self.near_thresholds = near_thresholds or NEAR_THRESHOLDS
        self.far_thresholds = far_thresholds or FAR_THRESHOLDS

        # Create MeanIoU for each (threshold x near/far x radius/box)
        self.metrics = {}
        for t in near_thresholds:
            for mode in ['radius', 'box']:
                key = f'near_{mode}_{t}'
                self.metrics[key] = MeanIoU(
                    list(range(1, 17)), EMPTY_LABEL, LABEL_STR, True,
                    EMPTY_LABEL, filter_minmax=False, name=key)
                self.metrics[key].reset()
        for t in far_thresholds:
            for mode in ['radius', 'box']:
                key = f'far_{mode}_{t}'
                self.metrics[key] = MeanIoU(
                    list(range(1, 17)), EMPTY_LABEL, LABEL_STR, True,
                    EMPTY_LABEL, filter_minmax=False, name=key)
                self.metrics[key].reset()

        # Distance bin accumulators
        self.dist_bins = np.arange(0, max_range + bin_size, bin_size)
        num_bins = len(self.dist_bins) - 1
        self.radial_nonempty = np.zeros(num_bins, dtype=np.int64)
        self.radial_total = np.zeros(num_bins, dtype=np.int64)
        self.box_nonempty = np.zeros(num_bins, dtype=np.int64)
        self.box_total = np.zeros(num_bins, dtype=np.int64)

    def update(self, pred_occ, gt_occ, sampled_xyz, occ_mask):
        """Call once per sample with flattened pred/gt/mask and (N, 3) xyz."""
        x, y = sampled_xyz[:, 0], sampled_xyz[:, 1]
        radial_dist = torch.sqrt(x**2 + y**2)
        box_dist = torch.max(x.abs(), y.abs())

        # Near thresholds
        for t in self.near_thresholds:
            nr_mask = (radial_dist <= t) & occ_mask
            nb_mask = (x.abs() <= t) & (y.abs() <= t) & occ_mask
            self.metrics[f'near_radius_{t}']._after_step(
                pred_occ[nr_mask], gt_occ[nr_mask])
            self.metrics[f'near_box_{t}']._after_step(
                pred_occ[nb_mask], gt_occ[nb_mask])

        # Far thresholds
        for t in self.far_thresholds:
            fr_mask = (radial_dist >= t) & occ_mask
            fb_mask = ((x.abs() >= t) | (y.abs() >= t)) & occ_mask
            self.metrics[f'far_radius_{t}']._after_step(
                pred_occ[fr_mask], gt_occ[fr_mask])
            self.metrics[f'far_box_{t}']._after_step(
                pred_occ[fb_mask], gt_occ[fb_mask])

        # Distance bin counts
        num_bins = len(self.dist_bins) - 1
        for b in range(num_bins):
            lo, hi = self.dist_bins[b], self.dist_bins[b + 1]
            r_mask = (radial_dist >= lo) & (radial_dist < hi) & occ_mask
            b_mask = (box_dist >= lo) & (box_dist < hi) & occ_mask
            self.radial_total[b] += r_mask.sum().item()
            self.radial_nonempty[b] += ((gt_occ != EMPTY_LABEL) & r_mask).sum().item()
            self.box_total[b] += b_mask.sum().item()
            self.box_nonempty[b] += ((gt_occ != EMPTY_LABEL) & b_mask).sum().item()

    def report(self, logger):
        """Log all spatial mIoU results and distance bin tables."""
        # Spatial mIoU for all thresholds
        for key in sorted(self.metrics.keys()):
            logger.info(f'\n===== {key} =====')
            s_miou, s_iou2 = self.metrics[key]._after_epoch()
            logger.info(f'{key} mIoU: {s_miou}, iou2: {s_iou2}')
            self.metrics[key].reset()

        # DDP reduction for bin counters
        if dist.is_initialized():
            for arr in [self.radial_nonempty, self.radial_total,
                        self.box_nonempty, self.box_total]:
                t = torch.tensor(arr, dtype=torch.long).cuda()
                dist.all_reduce(t)
                arr[:] = t.cpu().numpy()

        # Bin tables
        num_bins = len(self.dist_bins) - 1
        header = f'{"Bin (m)":>12} {"Total":>10} {"Non-empty":>10} {"% Non-empty":>12} {"% of all Non-empty":>18}'

        logger.info('\n===== Non-empty voxel distribution by RADIAL distance (sqrt(x^2+y^2)) =====')
        logger.info(header)
        radial_sum = self.radial_nonempty.sum()
        for b in range(num_bins):
            pct = 100.0 * self.radial_nonempty[b] / self.radial_total[b] if self.radial_total[b] > 0 else 0
            pct_tot = 100.0 * self.radial_nonempty[b] / radial_sum if radial_sum > 0 else 0
            logger.info(f'{self.dist_bins[b]:>5.0f}-{self.dist_bins[b+1]:>4.0f}m '
                        f'{self.radial_total[b]:>10d} {self.radial_nonempty[b]:>10d} '
                        f'{pct:>11.1f}% {pct_tot:>17.1f}%')

        logger.info('\n===== Non-empty voxel distribution by BOX distance (max(|x|,|y|)) =====')
        logger.info(header)
        box_sum = self.box_nonempty.sum()
        for b in range(num_bins):
            pct = 100.0 * self.box_nonempty[b] / self.box_total[b] if self.box_total[b] > 0 else 0
            pct_tot = 100.0 * self.box_nonempty[b] / box_sum if box_sum > 0 else 0
            logger.info(f'{self.dist_bins[b]:>5.0f}-{self.dist_bins[b+1]:>4.0f}m '
                        f'{self.box_total[b]:>10d} {self.box_nonempty[b]:>10d} '
                        f'{pct:>11.1f}% {pct_tot:>17.1f}%')
