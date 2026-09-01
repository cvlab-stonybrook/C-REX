import os
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import load_model, threshold
import os
import numpy as np
from datetime import datetime

from utils.processor import DataProcessor
from utils.criterion import SetCriterion, L2Loss, SetRegContrastiveCriterion
from utils.image_loader import get_loader
from tqdm import tqdm
from utils.util import visualize_and_save_points, visualize_density_map

from utils.tester_rec import *
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse

parser = argparse.ArgumentParser()
## training setting
parser.add_argument("--epochs", default=1, type=int, help="epoches")
parser.add_argument("--un_epochs", default=15, type=int, help="epoches")
parser.add_argument("--batch", default=1, type=int, help="batch size")
parser.add_argument("--seed", default=60, type=int, help="batch size")
parser.add_argument("--scale", default=1000, type=int, help="batch size")

## model setting 
parser.add_argument('--config', default="./configs/GroundingDINO_SwinB_density_guide.py", type=str, help='pretrain pth')
parser.add_argument('--pretrain_model', default="./exp/rec8k_crex_swinB/model.pth", type=str, help='pretrain pth')
parser.add_argument('--load_mode', default="test", type=str, help='loading mode')
parser.add_argument('--prompt_detach', action='store_true', help="if detach the prompt and density feature")

## saving setting
parser.add_argument("--stats_dir", default="./exp/test_aug/debug", type=str, help='stats directory')
parser.add_argument("--vis_dir", default="localization_test", type=str, help='stats directory')
parser.add_argument("--vis_density_dir", default="density", type=str, help='stats directory')
parser.add_argument("--result_txt", default="result.txt", type=str, help='stats directory')
parser.add_argument("--selection_txt", default="result_err_20_test.txt", type=str, help='stats directory')
parser.add_argument("--write_txt", action='store_true', help="write txt result")
parser.add_argument("--write_vis", action='store_true', help="write visual result")
parser.add_argument("--write_density", action='store_true', help="write density or not")
## test setting 
parser.add_argument("--pred_num_judge", default=650, type=int, help='patch threshold')
parser.add_argument("--threshold1", default=0.25, type=float, help='threshold for localization')
parser.add_argument("--threshold2", default=0.35, type=float, help='threshold for localization')

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

BATCH_SIZE = args.batch
train_loader = get_loader(processor, 'train', BATCH_SIZE)
val_loader = get_loader(processor, 'val', BATCH_SIZE)
test_loader = get_loader(processor, 'test', BATCH_SIZE)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")


""" model"""
CONFIG_PATH = args.config
CHECKPOINT_PATH =args.pretrain_model
print(f"Inference on test set using best model: {CHECKPOINT_PATH}")
model = load_model(CONFIG_PATH, CHECKPOINT_PATH, mode=args.load_mode)
model = model.to(device)
model.transformer.query_detach = args.prompt_detach

'''
这个是设置实验保存的位置，设置的是文件夹
'''
stats_dir = args.stats_dir
os.makedirs(stats_dir, exist_ok=True)

stats_file = f"{stats_dir}/stats.txt"
stats = list()

print(f"Saving stats to {stats_file}")

with open(stats_file, 'a') as f:
    header = ['train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN', 'train_precision', 'train_recall', 'train_f1', 'train_regression_mae','train_regression_rmse', 
              '||', 'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN', 'val_precision', 'val_recall', 'val_f1', 'val_regression_mae','val_regression_rmse',
              '||', 'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN', 'test_precision', 'test_recall', 'test_f1', 'test_regression_mae','test_regression_rmse']
    f.write("%s\n" % ' | '.join(header))
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression = eval('test', model, loaders, annotations, args)
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}, mae_regression: {test_mae_regression:5.2f}, mae_regression: {test_rmse_regression:5.2f}")
# write to stats file
line_inference = [0,0,0,0,0, 0, "||", 0,0,0,0,0, 0,"||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x,4))
    f.write("%s\n" % ' | '.join(s))



