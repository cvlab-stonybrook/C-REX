"""Run inference with a trained GroundingDINO (+ C-REX) checkpoint and dump per-(image, RE) predictions.

Example:
    python inference.py --dataset rec8k --config configs/main/rec8k_crex_swinT.py \
        --checkpoint runs/rec8k_crex_swinT/best_f1_model.pth --split test --output predictions.json
"""
import argparse
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from groundingdino.util.base_api import load_model, threshold
from utils.image_loader import get_loader

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    caption_sizes = [end_idx + 2 for end_idx in end_idxes]

    return [
        {"points": p.to(device), "labels": l.to(device), "caption_size": cs}
        for p, l, cs in zip(gt_points, gt_logits, caption_sizes)
    ]


def distance_threshold_func(boxes):
    if len(boxes) == 0:
        return 0.0
    areas = [box[2] * box[3] for box in boxes]
    median_box = boxes[np.argsort(areas)[len(areas) // 2]]
    return np.sqrt(median_box[2] ** 2 + median_box[3] ** 2) / 2.0


def calc_loc_metric(pred_boxes, gt_points):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rec8k", "fsc147"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", required=True, help="Path to write per-(image, RE) predictions as JSON")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--t1", type=float, default=0.25, help="CLS-token confidence threshold")
    parser.add_argument("--t2", type=float, default=0.35, help="Remaining-token confidence threshold")
    args = parser.parse_args()

    DataProcessor = get_processor_cls(args.dataset)
    processor = DataProcessor()
    annotations = processor.annotations
    loader = get_loader(processor, args.split, args.batch_size)
    print(f"{args.split}: {len(loader.dataset)} image-RE pairs")

    model = load_model(args.config, args.checkpoint).to(device)
    model.eval()

    predictions = {}
    mae_sum = rmse_sum = 0
    tp_sum = fp_sum = fn_sum = 0
    counter = 0

    for images, captions, shapes, img_caps in tqdm(loader):
        anno_b = [annotations[img_cap] for cap_list in img_caps for img_cap in cap_list]
        img_caps = [img_cap for cap_list in img_caps for img_cap in cap_list]
        shapes = [shapes[i] for i, cap_list in enumerate(captions) for _ in cap_list]

        images = torch.stack([images[i] for i, cap_list in enumerate(captions) for _ in cap_list], dim=0)
        captions = [cap for cap_list in captions for cap in cap_list]
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images, captions=captions)
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]

        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(anno_b, captions, shapes, emb_size, model.tokenizer)

        results = threshold(outputs, captions, model.tokenizer, args.t1, args.t2)
        for b in range(len(results)):
            boxes, logits, _ = results[b]
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

            img_id = img_caps[b][0]
            predictions.setdefault(img_id, {})[captions[b]] = {
                "points": points,
                "predicted": pred_cnt,
                "actual": gt_cnt,
                "abs_error": err,
                "TP": TP, "FP": FP, "FN": FN,
                "precision": precision, "recall": recall, "f1": f1,
            }

    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=4)

    mae = mae_sum / counter
    rmse = (rmse_sum / counter) ** 0.5
    precision = tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum else 0.0
    recall = tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{args.split} MAE: {mae:.2f} RMSE: {rmse:.2f} P: {precision:.2f} R: {recall:.2f} F1: {f1:.2f}")
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
