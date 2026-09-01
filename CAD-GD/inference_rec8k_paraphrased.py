import os
import sys
import json
import random

# Put the local GroundingDINO directory first so base_api and other custom
# files are found before any pip-installed groundingdino package.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "GroundingDINO"))
sys.path.insert(0, _REPO)

import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from groundingdino.util.base_api import load_model, threshold, load_image, preprocess_caption
from utils.processor import DataProcessor
from setproctitle import setproctitle
from tqdm import tqdm
import argparse

setproctitle("CAD-GD inference paraphrased")


# ── Paraphrase-aware DataProcessor ───────────────────────────────────────────

class DataProcessorParaphrased(DataProcessor):
    """Feeds paraphrased captions to the model; GT annotations unchanged."""
    def __init__(self):
        super().__init__()
        _map_path = os.path.join(
            os.path.dirname(_REPO),
            "referring-expression-counting", "anno", "re_paraphrase_map.json"
        )
        with open(_map_path) as f:
            self._para_map = json.load(f)

    def get_prompt_for_image(self, image_id_caption: tuple):
        _, caption = image_id_caption
        paraphrase = self._para_map.get(caption, caption)
        return [paraphrase]


# ── Density-free loader ───────────────────────────────────────────────────────

class Rec8KDatasetNoDensity(Dataset):
    def __init__(self, processor: DataProcessor, split):
        self.processor = processor
        split_set_tuples = processor.get_img_ids_for_split(split)
        split_dict = {}
        for img_id, cap in split_set_tuples:
            split_dict.setdefault(img_id, []).append(cap)
        self.img_ids = list(split_dict.keys())
        self.labels  = [list(split_dict[img_id]) for img_id in self.img_ids]
        self.img_cap_tuples = []
        for i, (img_id, caps) in enumerate(zip(self.img_ids, self.labels)):
            self.img_cap_tuples.append([(img_id, cap) for cap in caps])
            for j, cap in enumerate(caps):
                text_prompt = processor.get_prompt_for_image((img_id, cap))[0]
                self.labels[i][j] = preprocess_caption(caption=text_prompt)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.processor.get_image_path()
        img_file = os.path.join(img_path, self.img_ids[idx])
        image_source, image = load_image(img_file)
        h, w, _ = image_source.shape
        return image, self.labels[idx], (h, w), self.img_cap_tuples[idx]


def collate_fn(batch):
    images, labels, shapes, img_ids = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    padded = torch.zeros(len(images), 3, max_h, max_w)
    for i, img in enumerate(images):
        padded[i, :, :img.shape[1], :img.shape[2]] = img
    return padded, list(labels), list(shapes), list(img_ids)


def get_loader(processor, split, batch_size):
    ds = Rec8KDatasetNoDensity(processor, split)
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(split == "train"), collate_fn=collate_fn)


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--batch",          default=1,     type=int)
parser.add_argument("--seed",           default=314,   type=int)
parser.add_argument("--config",         default="/home_local/panosc/rec_obj/CAD-G./configs/GroundingDINO_SwinB_density_guide.py", type=str)
parser.add_argument("--load_mode",      default="test", type=str)
parser.add_argument("--prompt_detach",  action="store_true")
parser.add_argument("--model_path",     default="/nfs/bigtensor/add_disk1/panosc/continue/cadgd_crex_20_10_swinb/last_model.pth", type=str)
parser.add_argument("--threshold1",     default=0.25,  type=float)
parser.add_argument("--threshold2",     default=0.35,  type=float)
parser.add_argument("--out_name_val",   default="cadgd_paraphrased_val",  type=str)
parser.add_argument("--out_name_test",  default="cadgd_paraphrased_test", type=str)
args = parser.parse_args()
print(args)


# ── Setup ─────────────────────────────────────────────────────────────────────

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor   = DataProcessorParaphrased()
annotations = processor.annotations  # keyed on (img_id, original_re) — GT unchanged

BATCH_SIZE   = args.batch
train_loader = get_loader(processor, "train", BATCH_SIZE)
val_loader   = get_loader(processor, "val",   BATCH_SIZE)
test_loader  = get_loader(processor, "test",  BATCH_SIZE)
loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")


# ── Model ─────────────────────────────────────────────────────────────────────

print(f"Loading model from: {args.model_path}")
model = load_model(args.config, args.model_path, mode=args.load_mode)
model = model.to(device)
model.transformer.query_detach = args.prompt_detach

# Inject paraphrased REs into model.anno so split_caption can look them up.
_map_path = os.path.join(
    os.path.dirname(_REPO),
    "referring-expression-counting", "anno", "re_paraphrase_map.json"
)
with open(_map_path) as f:
    _para_map = json.load(f)
for orig_re, para_re in _para_map.items():
    if orig_re in model.anno and para_re not in model.anno:
        model.anno[para_re] = model.anno[orig_re]

for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.bert.parameters():
    param.requires_grad = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare_targets(anno_b, captions, shapes, emb_size):
    for anno in anno_b:
        if len(anno["points"]) == 0:
            anno["points"] = [[0, 0]]
    gt_points_b = [np.array(anno["points"]) / np.array(shape)[::-1]
                   for anno, shape in zip(anno_b, shapes)]
    gt_points = [torch.from_numpy(p).to(torch.float32) for p in gt_points_b]
    gt_logits = [torch.zeros((p.shape[0], emb_size)) for p in gt_points]

    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")
    end_idxes = [torch.where(ids == 1012)[0][-1] for ids in tokenized["input_ids"]]
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:, :end_idx] = 1.0
    caption_sizes = [end_idx + 2 for end_idx in end_idxes]

    return [{"points": p.to(device), "labels": l.to(device), "caption_size": cs}
            for p, l, cs in zip(gt_points, gt_logits, caption_sizes)]


def distance_threshold_func(boxes):
    if len(boxes) == 0:
        return 0.0
    areas = [box[2] * box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas) // 2]
    w, h = boxes[median_idx][2], boxes[median_idx][3]
    return np.sqrt(w**2 + h**2) / 2.0


def calc_loc_metric(pred_boxes, gt_points):
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_points), 0, 0, 0
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_pts = np.array([[b[0], b[1]] for b in pred_boxes])
    gt_pts   = gt_points.cpu().detach().numpy()
    cost     = cdist(pred_pts, gt_pts, metric="euclidean")
    pi, gi   = linear_sum_assignment(cost)
    TP = sum(1 for p, g in zip(pi, gi) if cost[p, g] < dist_threshold)
    FP = len(pred_pts) - TP
    FN = len(gt_pts)   - TP
    P  = TP / (TP + FP) if TP + FP else 0.0
    R  = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return TP, FP, FN, P, R, F1


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval(split, out_json):
    print(f"Evaluation on {split} set  (t1={args.threshold1:.2f}, t2={args.threshold2:.2f})")
    model.eval()
    loader = loaders[split]

    eval_mae = eval_rmse = 0
    eval_tp = eval_fp = eval_fn = 0
    counter = 0
    predictions = {}

    for images, captions, shapes, img_caps in tqdm(loader):
        anno_b   = [annotations[ic] for ic_list in img_caps for ic in ic_list]
        img_caps = [ic for ic_list in img_caps for ic in ic_list]
        shapes   = [shapes[i] for i, cl in enumerate(captions) for _ in cl]
        images   = torch.stack([images[i] for i, cl in enumerate(captions) for _ in cl], dim=0)
        captions = [cap for cl in captions for cap in cl]

        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=captions)

        emb_size = outputs["pred_logits"].shape[2]
        targets  = prepare_targets(anno_b, captions, shapes, emb_size)

        results = threshold(outputs, captions, model.tokenizer,
                            text_threshold=0,
                            threshold1=args.threshold1,
                            threshold2=args.threshold2)

        for b, (boxes, logits, _) in enumerate(results):
            boxes  = [box.tolist() for box in boxes]
            points = [[box[0], box[1]] for box in boxes]

            pred_cnt = len(points)
            gt_cnt   = len(targets[b]["points"])
            err      = abs(pred_cnt - gt_cnt)
            eval_mae  += err
            eval_rmse += err ** 2

            TP, FP, FN, P, R, F1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP;  eval_fp += FP;  eval_fn += FN
            counter += 1

            img_id, orig_re = img_caps[b]
            predictions.setdefault(img_id, {}).setdefault(captions[b], {})
            predictions[img_id][captions[b]] = {
                "original_re": orig_re,
                "points":      points,
                "predicted":   pred_cnt,
                "actual":      gt_cnt,
                "abs_error":   err,
            }

    with open(out_json, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved → {out_json}")

    mae  = eval_mae / counter
    rmse = (eval_rmse / counter) ** 0.5
    prec = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp else 0.0
    rec  = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return mae, rmse, eval_tp, eval_fp, eval_fn, prec, rec, f1


# ── Main ──────────────────────────────────────────────────────────────────────

"""val_mae, val_rmse, val_TP, val_FP, val_FN, val_prec, val_rec, val_f1 = \
    eval("val", f"{args.out_name_val}.json")
print(f"val  MAE: {val_mae:5.2f}, RMSE: {val_rmse:5.2f}, "
      f"TP: {val_TP}, FP: {val_FP}, FN: {val_FN}, "
      f"precision: {val_prec:5.2f}, recall: {val_rec:5.2f}, f1: {val_f1:5.2f}")"""

test_mae, test_rmse, test_TP, test_FP, test_FN, test_prec, test_rec, test_f1 = \
    eval("test", f"{args.out_name_test}.json")
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, "
      f"TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, "
      f"precision: {test_prec:5.2f}, recall: {test_rec:5.2f}, f1: {test_f1:5.2f}")
