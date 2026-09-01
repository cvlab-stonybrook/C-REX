import os
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import load_model, threshold, threshold_box
import os
import numpy as np
from datetime import datetime

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.criterion_box import SetCriterionBox, SetCriterionFSC147
from utils.image_loader import get_loader
from utils.image_loader_fsc147 import get_fsc_loader
from tqdm import tqdm
from utils.util import visualize_and_save_points, visualize_density_map
from utils.tester_fsc import *
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse

parser = argparse.ArgumentParser()
## training setting
parser.add_argument("--epochs", default=3, type=int, help="epoches")
parser.add_argument("--un_epochs", default=15, type=int, help="epoches")
parser.add_argument("--batch", default=4, type=int, help="batch size")
parser.add_argument("--seed", default=314, type=int, help="batch size")
parser.add_argument("--scale", default=1000, type=int, help="batch size")

parser.add_argument("--lr", default=1e-5, type=float, help="init lr")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="weight decay")
parser.add_argument("--localization_weight", default=1., type=float, help="localization weight")
parser.add_argument("--regression_weight", default=1., type=float, help="regression weight")
parser.add_argument("--step_size", default=10, type=int, help="step size for StepLR")
parser.add_argument("--density_warmup", action='store_true', help="warm up for density feature")
parser.add_argument("--warmup_ignore_localization", action='store_true', help="warm up for density feature")
parser.add_argument("--warmup_epoch", default=1, type=int, help="warm up epoch")

## model setting 
parser.add_argument('--config', default="./configs/GroundingDINO_SwinB_density_guide.py", type=str, help='pretrain pth')
parser.add_argument('--pretrain_model', default="./exp/fsc147_crex_swinB/model.pth", type=str, help='pretrain pth')
parser.add_argument('--load_mode', default="val", type=str, help='loading mode')
parser.add_argument('--prompt_detach', action='store_true', help="if detach the prompt and density feature")

## saving setting
parser.add_argument("--stats_dir", default="./exp/fsc147_debug/fsc_147_debug", type=str, help='stats directory')

## test setting 
parser.add_argument("--pred_num_judge", default=700, type=int, help='patch threshold')
parser.add_argument("--threshold", default=0.35, type=float, help='threshold for localization')

## data aug
parser.add_argument("--horizon_flip",  action='store_true', help="if horizon flip")
parser.add_argument("--horizon_flip_prob", default=0.5, type=float, help='probality for horizon flip')
parser.add_argument("--vertical_flip",  action='store_false', help="if vertical flip")
parser.add_argument("--vertical_flip_prob", default=0.5, type=float, help='probality for vertical flip')

args = parser.parse_args(args=[])
print(args)
""" seed fix """
seed_value = args.seed
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

""" data """
processor = DataProcessor()
annotations = processor.annotations

BATCH_SIZE_FSC = args.batch
train_loader = get_fsc_loader('train', BATCH_SIZE_FSC, args)
val_loader = get_fsc_loader('val', 1, args)
test_loader = get_fsc_loader('test', 1, args)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

stats_dir = args.stats_dir
os.makedirs(stats_dir, exist_ok=True)

stats_file = f"{stats_dir}/test_stats.txt"
stats = list()

print(f"Saving stats to {stats_file}")

with open(stats_file, 'a') as f:
    header = ['train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN', 'train_precision', 'train_recall', 'train_f1', 'train_regression_mae','train_regression_rmse', 
              '||', 'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN', 'val_precision', 'val_recall', 'val_f1', 'val_regression_mae','val_regression_rmse',
              '||', 'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN', 'test_precision', 'test_recall', 'test_f1', 'test_regression_mae','test_regression_rmse']
    f.write("%s\n" % ' | '.join(header))

""" model"""
CONFIG_PATH = args.config
CHECKPOINT_PATH = args.pretrain_model
model = load_model(CONFIG_PATH, CHECKPOINT_PATH, mode='test')
model = model.to(device)

# test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression = eval('test', model, loaders, args)
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1 = eval_reproduce(args.load_mode, model, loaders, args)
# print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}, mae_regression: {test_mae_regression:5.2f}, mae_regression: {test_rmse_regression:5.2f}")
# # write to stats file
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}")
# write to stats file
# line_inference = [0,0,0,0,0, 0,0,0, "||", 0,0,0,0,0, 0,0,0, "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression]
line_inference = [0,0,0,0,0, 0,0,0, "||", 0,0,0,0,0, 0,0,0, "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1,]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x,4))
    f.write("%s\n" % ' | '.join(s))
