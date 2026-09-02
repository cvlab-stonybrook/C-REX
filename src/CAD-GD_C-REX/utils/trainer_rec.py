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
import torchvision.transforms.functional as TF

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_localization = SetCriterion()
criterion_counting = L2Loss()

def train(model, loaders, optimizer, scheduler, annotations, args):
    print(f"Training on train set data")
    model.train()
    loader = loaders['train']

    train_mae = 0
    train_rmse = 0
    train_mae_regression = 0
    train_rmse_regression = 0
    
    train_tp = 0
    train_fp = 0
    train_fn = 0
    
    counter = 0
    counter_for_image = 0
    train_size = len(loader.dataset) 

    loss_regression_sum = 0.
    loss_localization_sum = 0.
    loss_regression_contrastive_sum = 0.
    
    total_num = 0
    my_count_brother = 0
    for images, captions, shapes, img_caps, density_maps in tqdm(loader): # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch

        mask_bi = [i for i, img_cap_list in enumerate(img_caps) for _ in img_cap_list] # index for each img,cap pair in the batch
        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]
        
        optimizer.zero_grad()

        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)

        outputs = model(images, captions=captions)
        # density map and loss
        pred_density_map = outputs['density'].squeeze(1)
        density_maps = density_maps.to(pred_density_map.device)
        density_maps_gt = density_maps * args.scale
        loss_regression = criterion_counting(pred_density_map, density_maps_gt)
        emb_size = outputs["pred_logits"].shape[2]
        # mae 
        # pred_num 
        targets = prepare_targets(model, anno_b, captions, shapes, emb_size)

        pred_num = torch.sum(pred_density_map, dim=[1,2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim = [1,2])
        for i, p in enumerate(targets):
            # gt_num = len(p['points'])
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            train_mae_regression += cnt_err
            train_rmse_regression += cnt_err ** 2

        total_num += density_maps.shape[0]

        # localization target and loss
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
        
        # prepare targets

        loss_dict = criterion_localization(outputs, targets, mask_bi)
        weight_dict = criterion_localization.weight_dict
        # loss_regression_contrastive = criterion_contrastive_regression(outputs, targets, mask_bi)
        loss_localization = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # loss = loss_localization + loss_regression + 0.04 * loss_regression_contrastive
        loss = args.localization_weight * loss_localization + args.regression_weight * loss_regression
        loss_regression_sum += loss_regression.item()
        loss_localization_sum += loss_localization.item()
        # loss_regression_contrastive_sum += loss_regression_contrastive.item()
        loss.backward()
        optimizer.step()
        counter_for_image += 1
        results = threshold(outputs, captions, model.tokenizer, text_threshold=0, threshold1=args.threshold1, threshold2=args.threshold2)
        for b in range(len(results)): # (bs*num_cap)
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            points = [[box[0], box[1]] for box in boxes] # center points

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            train_mae += cnt_err
            train_rmse += cnt_err ** 2

            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            train_tp += TP
            train_fp += FP
            train_fn += FN
        
            counter += 1
            
            # print(f'[train] ep {epoch} ({counter_for_image}/{train_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(train_mae/counter)}, RMSE: {int((train_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
        
        if counter_for_image % 5 == 0:
            print(f"lr:{optimizer.param_groups[0]['lr']}",
                  f'loss_regression:{round(loss_regression_sum/counter_for_image,7)}', 
                  f'mae_regression:{round(train_mae_regression.item()/total_num,2)}',
                  f'mse_regression: {round((train_rmse_regression.item()/total_num)**0.5,2)}',
                  f'loss_localization:{round(loss_localization_sum/counter_for_image,5)}',
                  f'mae:{round(train_mae/total_num,2)}',
                  f'mse:{round((train_rmse/total_num)**0.5,2)}')
            
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    train_regression_mae = train_mae_regression.item() / counter
    train_regression_rmse = (train_rmse_regression.item() / counter) ** 0.5

    train_mae = train_mae / counter
    train_rmse = (train_rmse / counter) ** 0.5

    train_precision = train_tp / (train_tp + train_fp) if train_tp + train_fp != 0 else 0.0
    train_recall = train_tp / (train_tp + train_fn) if train_tp + train_fn != 0 else 0.0
    train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if train_precision + train_recall != 0 else 0.0

    return train_mae, train_rmse, train_tp, train_fp, train_fn, train_precision, train_recall, train_f1, train_regression_mae, train_regression_rmse


def eval_fn(split, model, loaders, annotations, args):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

    eval_mae = 0
    eval_rmse = 0

    val_mae = 0
    val_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0
    
    counter = 0
    counter_for_image = 0
    eval_size = len(loader.dataset)

    total_num = 0

    for images, captions, shapes, img_caps, density_maps in loader: # tensor, list, list, list

        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]
        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=captions)
        ## density output
        pred_density_map = outputs['density'].squeeze(1)
        density_maps = density_maps.to(pred_density_map.device)
        ### mae 
        total_num += density_maps.shape[0]
        counter_for_image += 1

        ## localization output
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(model, anno_b, captions, shapes, emb_size)
        # pred_num 
        pred_num = torch.sum(pred_density_map, dim=[1,2]) / args.scale
        gt_num_density = torch.sum(density_maps, dim = [1,2])
        for i, p in enumerate(targets):
            # gt_num = len(p['points'])
            cnt_err = torch.abs(pred_num[i] - gt_num_density[i])
            val_mae += cnt_err
            val_rmse += cnt_err ** 2

        
        results = threshold(outputs, captions, model.tokenizer, text_threshold=0, threshold1=args.threshold1, threshold2=args.threshold2)
        for b in range(len(results)):
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()


            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2
        
            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN


            counter += 1
            
            # print(f'[{split}] ep {epoch} ({counter_for_image}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(eval_mae/counter)}, RMSE: {int((eval_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
        
    eval_mae_regression = round(val_mae.item()/total_num,2)
    eval_rmse_regression = round(val_rmse.item()/total_num,2)**0.5

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1, eval_mae_regression, eval_rmse_regression


def prepare_targets(model, anno_b, captions, shapes, emb_size):
    for anno in anno_b:
        if len(anno['points']) == 0:
            anno['points'] = [[0,0]]
    gt_points_b = [np.array(anno['points']) / np.array(shape)[::-1] for anno, shape in zip(anno_b, shapes)] # (h,w) -> (w,h)
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b] 

    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points] 

    
    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")

    # find last index of special token (.)
    end_idxes = [torch.where(input_ids==1012)[0][-1] for input_ids in tokenized['input_ids']] 
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:,:end_idx] = 1.0 

    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [{"points": img_gt_points.to(device), "labels": img_gt_logits.to(device), "caption_size": caption_size} for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)] 

    return targets


def distance_threshold_func(boxes): # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2]*box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas)//2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0
    
    return threshold

def calc_loc_metric(pred_boxes, gt_points): # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0
    
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1
    
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0
    
    return TP, FP, FN, Precision, Recall, F1
