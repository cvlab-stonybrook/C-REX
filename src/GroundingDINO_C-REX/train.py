"""Train/evaluate GroundingDINO (+ C-REX) on REC-8K or FSC-147.

Example:
    python train.py --dataset rec8k --config configs/main/rec8k_crex_swinT.py \
        --checkpoint groundingdino_swint_ogc.pth --output-dir runs/rec8k_crex_swinT

    python train.py --dataset fsc147 --config configs/main/fsc147_crex_swinT.py \
        --checkpoint groundingdino_swint_ogc.pth --output-dir runs/fsc147_crex_swinT --batch-size 2
"""
import argparse
import copy
import os
import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from groundingdino.util.base_api import load_model, threshold
from groundingdino.util.slconfig import SLConfig
from utils.criterion import SetCriterion
from utils.image_loader import get_loader

device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_processor_cls(dataset):
    if dataset == "rec8k":
        from utils.processor import DataProcessor
    elif dataset == "fsc147":
        from utils.processor_fsc import DataProcessor
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DataProcessor


def prepare_targets(anno_b, captions, shapes, emb_size, tokenizer):
    for anno in anno_b:
        if len(anno["points"]) == 0:
            anno["points"] = [[0, 0]]
    gt_points_b = [np.array(anno["points"]) / np.array(shape)[::-1] for anno, shape in zip(anno_b, shapes)]
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b]
    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points]

    tokenized = tokenizer(captions, padding="longest", return_tensors="pt")
    end_idxes = [torch.where(input_ids == 1012)[0][-1] for input_ids in tokenized["input_ids"]]
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:, :end_idx] = 1.0
    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [
        {"points": p.to(device), "labels": l.to(device), "caption_size": cs}
        for p, l, cs in zip(gt_points, gt_logits, caption_sizes)
    ]
    return targets


def distance_threshold_func(boxes):  # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    areas = [box[2] * box[3] for box in boxes]
    median_box = boxes[np.argsort(areas)[len(areas) // 2]]
    return np.sqrt(median_box[2] ** 2 + median_box[3] ** 2) / 2.0


def calc_loc_metric(pred_boxes, gt_points):  # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_points), 0, 0, 0

    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    cost_matrix = cdist(pred_points, gt_points, metric="euclidean")
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    TP = sum(1 for p, g in zip(pred_indices, gt_indices) if cost_matrix[p, g] < dist_threshold)
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TP, FP, FN, precision, recall, f1


def run_epoch(model, criterion, loader, annotations, optimizer=None, log_every=500):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    mae_sum = rmse_sum = 0
    tp_sum = fp_sum = fn_sum = 0
    counter = 0
    counter_for_image = 0
    total = len(loader.dataset)

    for images, captions, shapes, img_caps in loader:
        mask_bi = [i for i, cap_list in enumerate(img_caps) for _ in cap_list]
        anno_b = [annotations[img_cap] for cap_list in img_caps for img_cap in cap_list]
        img_caps = [img_cap for cap_list in img_caps for img_cap in cap_list]
        shapes = [shapes[i] for i, cap_list in enumerate(captions) for _ in cap_list]

        images = torch.stack([images[i] for i, cap_list in enumerate(captions) for _ in cap_list], dim=0)
        captions = [cap for cap_list in captions for cap in cap_list]
        images = images.to(device)

        if train_mode:
            optimizer.zero_grad()
            outputs = model(images, captions=captions)
        else:
            with torch.no_grad():
                outputs = model(images, captions=captions)

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(anno_b, captions, shapes, emb_size, model.tokenizer)

        if train_mode:
            loss_dict = criterion(outputs, targets, mask_bi)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            loss.backward()
            optimizer.step()

        counter_for_image += 1
        results = threshold(outputs, captions, model.tokenizer)
        for b in range(len(results)):
            boxes, _, _ = results[b]
            boxes = [box.tolist() for box in boxes]
            points = [[box[0], box[1]] for box in boxes]

            pred_cnt, gt_cnt = len(points), len(targets[b]["points"])
            err = abs(pred_cnt - gt_cnt)
            mae_sum += err
            rmse_sum += err ** 2

            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            tp_sum += TP
            fp_sum += FP
            fn_sum += FN
            counter += 1

            if counter_for_image % log_every == 0:
                phase = "train" if train_mode else "eval"
                print(
                    f"[{phase}] ({counter_for_image}/{total}) {img_caps[b]!r} / {captions[b]!r}: "
                    f"gt={gt_cnt} pred={pred_cnt} | MAE={mae_sum / counter:.2f} "
                    f"RMSE={(rmse_sum / counter) ** 0.5:.2f} P={precision:.2f} R={recall:.2f} F1={f1:.2f}"
                )

    mae = mae_sum / counter
    rmse = (rmse_sum / counter) ** 0.5
    precision = tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return mae, rmse, tp_sum, fp_sum, fn_sum, precision, recall, f1


def format_row(values):
    return " | ".join(str(round(v, 4)) if not isinstance(v, str) else v for v in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rec8k", "fsc147"], required=True)
    parser.add_argument("--config", required=True, help="Path to a GroundingDINO config .py file")
    parser.add_argument("--checkpoint", required=True, help="Pretrained GroundingDINO checkpoint to initialize from")
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints and stats.txt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["plateau", "steplr"], default=None,
                         help="Default: 'plateau' for rec8k, 'steplr' for fsc147")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume model weights from")
    parser.add_argument("--start-epoch", type=int, default=0)
    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    stats_file = os.path.join(args.output_dir, "stats.txt")

    DataProcessor = get_processor_cls(args.dataset)
    processor = DataProcessor()
    annotations = processor.annotations

    loaders = {
        split: get_loader(processor, split, args.batch_size) for split in ("train", "val", "test")
    }
    print("Data loaded!")
    print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} | Test: {len(loaders['test'].dataset)}")

    model = load_model(args.config, args.checkpoint).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.bert.parameters():
        param.requires_grad = False

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["model"])

    cfg = SLConfig.fromfile(args.config)
    criterion = SetCriterion(cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_kind = args.scheduler or ("steplr" if args.dataset == "fsc147" else "plateau")
    if scheduler_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=7, min_lr=1e-7, threshold=2e-3)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best = {"f1": 0.0, "mae": float("inf"), "rmse": float("inf")}
    best_test = {"f1": 0.0, "mae": float("inf"), "rmse": float("inf")}
    best_paths = {}

    header = [
        "train_mae", "train_rmse", "train_TP", "train_FP", "train_FN", "train_precision", "train_recall", "train_f1",
        "||", "val_mae", "val_rmse", "val_TP", "val_FP", "val_FN", "val_precision", "val_recall", "val_f1",
        "||", "test_mae", "test_rmse", "test_TP", "test_FP", "test_FN", "test_precision", "test_recall", "test_f1",
    ]
    with open(stats_file, "a") as f:
        f.write(format_row(header) + "\n")
    print(f"Saving stats to {stats_file}")

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = run_epoch(model, criterion, loaders["train"], annotations, optimizer=optimizer)
        val_stats = run_epoch(model, criterion, loaders["val"], annotations)
        test_stats = run_epoch(model, criterion, loaders["test"], annotations)

        val_f1, val_mae, val_rmse = val_stats[7], val_stats[0], val_stats[1]
        test_f1, test_mae, test_rmse = test_stats[7], test_stats[0], test_stats[1]

        if scheduler_kind == "plateau":
            scheduler.step(val_f1)
        else:
            scheduler.step()

        for metric, val, is_better in (
            ("f1", val_f1, val_f1 > best["f1"]),
            ("mae", val_mae, val_mae < best["mae"]),
            ("rmse", val_rmse, val_rmse < best["rmse"]),
        ):
            if is_better:
                best[metric] = val
                path = os.path.join(args.output_dir, f"best_{metric}_model.pth")
                torch.save({"model": copy.deepcopy(model.state_dict())}, path)
                best_paths[metric] = path
                print(f"New best val {metric}: {val:.4f}")

        for metric, val, is_better in (
            ("f1", test_f1, test_f1 > best_test["f1"]),
            ("mae", test_mae, test_mae < best_test["mae"]),
            ("rmse", test_rmse, test_rmse < best_test["rmse"]),
        ):
            if is_better:
                best_test[metric] = val
                path = os.path.join(args.output_dir, f"best_{metric}_model_test.pth")
                torch.save({"model": copy.deepcopy(model.state_dict())}, path)
                best_paths[f"{metric}_test"] = path

        row = list(train_stats) + ["||"] + list(val_stats) + ["||"] + list(test_stats)
        with open(stats_file, "a") as f:
            f.write(format_row(row) + "\n")

    torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "final_model.pth"))
    torch.save({"optimizer": optimizer.state_dict()}, os.path.join(args.output_dir, "final_optimizer.pth"))

    # Final test-set evaluation with each of the best validation checkpoints
    for metric in ("f1", "mae", "rmse"):
        path = best_paths.get(metric)
        if path is None:
            continue
        print(f"Inference on test set using best val {metric} model: {path}")
        model = load_model(args.config, path).to(device)
        test_stats = run_epoch(model, criterion, loaders["test"], annotations)
        print(f"test MAE: {test_stats[0]:.2f} RMSE: {test_stats[1]:.2f} F1: {test_stats[7]:.2f}")
        row = [0] * 8 + ["||"] + [0] * 8 + ["||"] + list(test_stats)
        with open(stats_file, "a") as f:
            f.write(format_row(row) + "\n")


if __name__ == "__main__":
    main()
