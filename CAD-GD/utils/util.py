import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm

def attention_map_save(attention_map, pth):
    to_pil = transforms.ToPILImage()
    attention_map = (attention_map - torch.min(attention_map)) / (torch.max(attention_map) - torch.min(attention_map))
    a = to_pil(attention_map)
    plt.imshow(a, cmap='jet') 
    plt.colorbar()  
    plt.savefig(pth)
    plt.close()

def channel_map_save(channel_attention, pth):
    channel_attention = (channel_attention - torch.min(channel_attention)) / (torch.max(channel_attention) - torch.min(channel_attention))
    plt.imshow(channel_attention.numpy(), cmap='hot', aspect='auto')
    plt.title("1x256 Vector Visualization (Heatmap)")
    plt.colorbar()  

    plt.savefig(pth)
    plt.close()

def density_map_save(pth, density_map):
    density = (density_map - torch.min(density_map)) / (torch.max(density_map) - torch.min(density_map))
    density = density.cpu().detach().numpy()
    cv2.imwrite(pth, density*255)

def image_save(pth, img):
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    img = img.permute(1,2,0).cpu().detach().numpy()
    cv2.imwrite(pth,img[:,:,[2,1,0]]*255)
    

def top_n_positions(tensor, n):

    b, h, w = tensor.shape
    flattened = tensor.view(b, -1)
    top_n_indices = flattened.topk(n, dim=1).indices  

    row_indices = top_n_indices // w  
    col_indices = top_n_indices % w   
    
    indices = torch.stack((row_indices, col_indices), dim=-1)
    
    return indices

def find_local_maxima(tensor, neighborhood_size=10):
    padding = neighborhood_size // 2
    padded_tensor = F.pad(tensor, (padding, padding, padding, padding), mode='replicate')
    max_pool = F.max_pool2d(padded_tensor, kernel_size=neighborhood_size, stride=1)
    max_pool = max_pool[:, :tensor.shape[1], :tensor.shape[2]]
    local_max_mask = (tensor == max_pool)

    return local_max_mask.int()


def visualize_and_save_points(image_tensor, gt_points, pred_points, save_path, label_info):
    pred_cnt, gt_cnt, TP, FP, FN = label_info
    
    plt.clf()
    image = np.transpose(image_tensor, (1, 2, 0))

    h, w = image.shape[:2]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    gt_points_abs = gt_points * np.array([w, h])
    pred_points_abs = pred_points * np.array([w, h])

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=800)

    ax.imshow(image)

    if len(gt_points_abs) > 0:
        ax.scatter(gt_points_abs[:, 0], gt_points_abs[:, 1], c=(210/255, 45/255, 0/255), marker='o', s=60)  # 红色点标记
    
    if len(pred_points_abs) > 0:
        ax.scatter(pred_points_abs[:, 0], pred_points_abs[:, 1], c=(0/255, 145/255, 210/255), marker='*',  s=60)  # 蓝色五角星，s用于调整大小

    ax.axis('off')
    plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_density_map(density_map, img=None, save_path=None):
    if img is not None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # 调整维度方便广播
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        img = img * std + mean

        if isinstance(img, torch.Tensor):
            img = img.numpy()  
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img, alpha=1)

    plt.imshow(density_map, cmap='viridis', alpha=0.6) 
    if save_path:
        plt.axis('off')
        plt.savefig(save_path, dpi=800,bbox_inches='tight', pad_inches=0)
        plt.close() 
    else:
        plt.show() 
        
def calculate_dynamic_threshold(predicted_count):
    b = predicted_count.shape
    thresholds = []
    for i in range(b[0]):
        if predicted_count[i] > 100:
            threshold = 0.32
        elif predicted_count[i] <= 100:
            threshold = 0.36
        thresholds.append(threshold)
    return thresholds

def plot_pca_features(features, b, save_pth):
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    

    colors = cm.rainbow(np.linspace(0, 1, b))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(b):

        group_features = features[i]

        pca = PCA(n_components=2)
        group_features_pca = pca.fit_transform(group_features)
 
        plt.scatter(group_features_pca[:, 0], group_features_pca[:, 1], 
                    color=colors[i], label=f'Group {i+1}', s=10, alpha=0.7)


    plt.title('PCA Feature Distribution of Different Groups')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig(save_pth, dpi=600)
    plt.close()
    
def plot_tsne_features_polar_colored(features, b, save_pth, captions, perplexity=30, learning_rate=200, n_iter=1000, seed=0):
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    colormaps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis]
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True) 
    
    for i in range(b):
        group_features = features[i]
        
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                    learning_rate=learning_rate, n_iter=n_iter, random_state=seed)
        group_features_tsne = tsne.fit_transform(group_features)
        
        angles = np.arctan2(group_features_tsne[:, 1], group_features_tsne[:, 0]) 
        radii = np.sqrt(group_features_tsne[:, 0]**2 + group_features_tsne[:, 1]**2) 
        
        colormap = colormaps[i % len(colormaps)]
        colors = colormap(radii / radii.max()) 

        ax.scatter(angles, radii, color=colors, label=captions[i], s=10, alpha=0.7)

    plt.title('t-SNE Feature Distribution of Different Groups in Polar Coordinates')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(save_pth, dpi=600)
    plt.show()

def plot_tsne_features_varying_radius(features, b, save_pth, captions, perplexity=30, learning_rate=200, n_iter=1000, seed=0,alpha=1,grid_size=100, max_radis=24, color_setting=0):
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    colormaps = [plt.cm.Purples, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)  
    ax.set_ylim(0, max_radis)
    for i in range(b):

        group_features = features[i]

        tsne = TSNE(n_components=2, perplexity=perplexity, 
                    learning_rate=learning_rate, n_iter=n_iter, random_state=seed)
        group_features_tsne = tsne.fit_transform(group_features)

        angles = np.arctan2(group_features_tsne[:, 1], group_features_tsne[:, 0])  
        radii = np.sqrt(group_features_tsne[:, 0]**2 + group_features_tsne[:, 1]**2)  

        theta_grid, r_grid = np.meshgrid(np.linspace(-np.pi, np.pi, grid_size), np.linspace(0, radii.max(), grid_size))
     
        density, _, _ = np.histogram2d(angles, radii, bins=[grid_size, grid_size], range=[[-np.pi, np.pi], [0, radii.max()]])
        density = density / density.max()  

        colormap = colormaps[color_setting]
        mesh = ax.pcolormesh(theta_grid, r_grid, density.T, cmap=colormap, shading='auto', alpha=alpha)
        plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.1, label=f'Density - {captions[i]}')

    plt.title('t-SNE Feature Distribution in Polar Coordinates with Varying Radii')
    plt.savefig(save_pth, dpi=600)
    plt.show()
    
def plot_tsne_features_pcolormesh(features, b, save_pth, captions, radius=10, perplexity=30, learning_rate=200, n_iter=1000, seed=0, grid_size=100):
    if isinstance(features, torch.Tensor):
        features = features.numpy()

    colormaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True) 

    for i in range(b):

        group_features = features[i]

        tsne = TSNE(n_components=2, perplexity=perplexity, 
                    learning_rate=learning_rate, n_iter=n_iter, random_state=seed)
        group_features_tsne = tsne.fit_transform(group_features)
        
        angles = np.arctan2(group_features_tsne[:, 1], group_features_tsne[:, 0])  # 角度
        fixed_radii = np.full_like(angles, radius)  

        theta_grid, r_grid = np.meshgrid(np.linspace(-np.pi, np.pi, grid_size), np.linspace(radius - 1, radius + 1, 2))

        density, _ = np.histogram(angles, bins=grid_size, range=(-np.pi, np.pi))
        density = density / density.max()  
        density_grid = np.tile(density, (2, 1))  
        

        colormap = colormaps[i % len(colormaps)]
        mesh = ax.pcolormesh(theta_grid, r_grid, density_grid, cmap=colormap, shading='auto', alpha=0.7)
        plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.1, label=f'Density - {captions[i]}')
    plt.title('t-SNE Feature Distribution in Polar Coordinates with Varying Radii')
    plt.savefig(save_pth, dpi=600)
    plt.show()
    
def plot_tsne_features(features, b, save_pth, captions, perplexity=30, learning_rate=200, n_iter=1000, seed=0):
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    

    colors = cm.rainbow(np.linspace(0, 1, b))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(b):
        group_features = features[i]
        

        tsne = TSNE(n_components=2, perplexity=perplexity, 
                    learning_rate=learning_rate, n_iter=n_iter, random_state=seed)
        group_features_tsne = tsne.fit_transform(group_features)
        

        plt.scatter(group_features_tsne[:, 0], group_features_tsne[:, 1], 
                    color=colors[i], label=captions[i], s=10, alpha=0.7)


    plt.title('t-SNE Feature Distribution of Different Groups')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig(save_pth, dpi=600)
    plt.show()
