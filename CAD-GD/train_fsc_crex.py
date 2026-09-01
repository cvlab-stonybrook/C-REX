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
from utils.criterion import SetCriterionCREX, L2Loss, SetRegContrastiveCriterion
from utils.image_loader import get_loader
from utils.image_loader_fsc147 import get_fsc_loader
from tqdm import tqdm
from utils.util import visualize_and_save_points, visualize_density_map
from utils.trainer_fsc_crex import *
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse

from setproctitle import setproctitle

setproctitle("CADGD FSC CREX swinT")

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
parser.add_argument('--config', default="./configs/FSC_GroundingDINO_SwinT_density_guide.py", type=str, help='pretrain pth')
parser.add_argument('--pretrain_model', default="./groundingdino_swint_ogc.pth", type=str, help='pretrain pth')
parser.add_argument('--load_mode', default="train", type=str, help='loading mode')
parser.add_argument('--prompt_detach', action='store_true', help="if detach the prompt and density feature")

## saving setting
parser.add_argument("--stats_dir", default="./exp/fsc147_debug/fsc_147_debug_1", type=str, help='stats directory')

## test setting 
parser.add_argument("--pred_num_judge", default=10000, type=int, help='patch threshold')
parser.add_argument("--threshold", default=0.3, type=float, help='threshold for localization')

## augmentation setting
parser.add_argument("--horizon_flip",  action='store_true', help="if horizon flip")
parser.add_argument("--horizon_flip_prob", default=0.5, type=float, help='probality for horizon flip')
parser.add_argument("--vertical_flip",  action='store_true', help="if vertical flip")
parser.add_argument("--vertical_flip_prob", default=0.5, type=float, help='probality for vertical flip')

args = parser.parse_args()
print(args)
""" seed fix """
seed_value = args.seed
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

""" data """
processor = DataProcessor(dataset="fsc")
annotations = processor.annotations

BATCH_SIZE_FSC = args.batch
train_loader = get_fsc_loader('train', BATCH_SIZE_FSC, args)
val_loader = get_fsc_loader('val', 1, args)
test_loader = get_fsc_loader('test', 1, args)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")


""" model"""
CONFIG_PATH = args.config
CHECKPOINT_PATH = args.pretrain_model
model = load_model(CONFIG_PATH, CHECKPOINT_PATH, mode=args.load_mode)
model = model.to(device)

model.transformer.query_detach = args.prompt_detach

# freeze encoders
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.bert.parameters():
    param.requires_grad = False


""" criterion """
# lr = optimizer.param_groups[0]['lr'] 查看当前的学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size)

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


best_f1 = -1
best_model = None
save_every_epoch = False

best_mae, best_test_mae = 1000, 1000

model.transformer.density_warmup = args.density_warmup

localization_weight = args.localization_weight
localization_regression = args.regression_weight

for epoch in range(0, args.epochs):
        
    train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1, train_mae_regression, train_rmse_regression = train(model, loaders, optimizer, scheduler, args)
    val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1, val_mae_regression, val_rmse_regression = eval_fn('val', model, loaders, args)
    test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression = eval_fn('test', model, loaders, args)
    
    if best_mae > val_mae:
        best_mae = val_mae
        print(f"New best mae: {best_mae}")
        best_model = copy.deepcopy(model)

    if best_test_mae > test_mae:
        best_test_mae = test_mae
        print(f"New best mae: {best_test_mae}")
        best_model_test = copy.deepcopy(model)
    
    if epoch % 10 == 0 and epoch>=10: 
        epoch_model = copy.deepcopy(model)
        model_name = f'{stats_dir}/model_{str(epoch)}.pth'
        torch.save({"model": epoch_model.state_dict()}, model_name)

    stats.append([train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1,train_mae_regression, train_rmse_regression,
                   "||", val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1,val_mae_regression, val_rmse_regression,
                     "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression])

    with open(stats_file, 'a') as f:
        s = stats[-1]
        for i, x in enumerate(s):
            if type(x) != str:
                s[i] = str(round(x,4))
        f.write("%s\n" % ' | '.join(s))

    last_model_name = f'{stats_dir}/last_model.pth'
    torch.save({"model": model.state_dict()}, last_model_name)

    model_name = f'{stats_dir}/model.pth'
    torch.save({"model": best_model.state_dict()}, model_name)

    model_name_test = f'{stats_dir}/test_model.pth'
    torch.save({"model": best_model_test.state_dict()}, model_name)


# Inference on test set
print(f"Inference on test set using best model: {model_name}")
model = load_model(CONFIG_PATH, model_name, mode='test')
model = model.to(device)
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression = eval('test', model, loaders, args)
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}, mae_regression: {test_mae_regression:5.2f}, mae_regression: {test_rmse_regression:5.2f}")
# write to stats file
line_inference = [0,0,0,0,0, 0,0,0, "||", 0,0,0,0,0, 0,0,0, "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1, test_mae_regression, test_rmse_regression]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x,4))
    f.write("%s\n" % ' | '.join(s))


