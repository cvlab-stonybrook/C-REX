import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import plot_tsne_features,plot_tsne_features_pcolormesh, plot_tsne_features_varying_radius
## 这种方式产生的query是对text进行适配的。
## 因此我们可以考虑在增加一个对于density的考虑。
class QueryLinguishModule(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 定义可学习参数矩阵
        self.learnable_matrix = nn.Parameter(torch.randn(c, c))

    def forward(self, query, text_features):
        # 将文本特征与可学习参数相乘, 对于给定文本是固定的。
        t_prime = torch.matmul(text_features, self.learnable_matrix)
        
        # 经过 GeLU 激活函数
        t_prime_activated = F.gelu(t_prime)
        
        # 将激活后的特征与 query 相乘
        # query: (b, 900, c), t_prime_activated: (b, n, c)
        # 需要调整维度以便相乘
        query_expanded = query.unsqueeze(2)  # (b, 900, 1, c)
        t_prime_expanded = t_prime_activated.unsqueeze(1)  # (b, 1, n, c)
        
        # 相乘并求和以得到权重图
        weight_map = torch.matmul(query_expanded, t_prime_expanded.transpose(-1, -2)).squeeze(2)  # (b, 900, n)
        
        # 输出文本增强的queries
        k = text_features.shape[1]
        query_with_linguish = torch.matmul(weight_map, text_features)
        return query_with_linguish


# class QueryDensityModule(nn.Module):
#     def __init__(self, c, channel_attention=True):
#         super().__init__()
#         # 定义可学习参数矩阵
#         self.spatial_module = SpatialModule()
#         self.channel_attention = channel_attention
#         if self.channel_attention:
#             self.channel_module = ChannelModule(c)

#     def forward(self, query, density_feature):
#         # 将文本特征与可学习参数相乘
#         query = self.spatial_module(query, density_feature)
#         if self.channel_attention:
#             query = self.channel_module(query)
#         return query

class QueryAttentionModule(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.cross_attention =CrossAttentionLayer(c)

    def forward(self, query, density_feature):
        query = self.cross_attention(query, density_feature)
        return query

### 将density feature加入到query当中。
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, density_feature):
        Q = query.transpose(0, 1)
        K = density_feature.transpose(0, 1)
        V = density_feature.transpose(0, 1)
        attn_output, _ = self.cross_attention(Q, K, V)
        # Add & norm
        updated_query = self.norm(attn_output.transpose(0, 1) + query)
        return updated_query

# class SpatialModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # get max pool and avg pool of every layer
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         # 
#         self.spatial_attention = nn.Linear(2, 1)
#         # self.spatial_attention =nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, query, density_feature):
#         density_max = self.max_pool(density_feature)
#         density_mean = self.avg_pool(density_feature)
#         spatial_feature = torch.cat([density_max, density_mean], dim=-1)
#         attention_spatial = self.sigmoid(self.spatial_attention(spatial_feature))
#         query = query * attention_spatial
#         return query

# class ChannelModule(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         ## Channel attention map
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)

#         self.mlp = MLP(c, c, c, num_layers=3)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, query):
#         max_x = self.mlp(self.max_pool(query.permute(0,2,1)).permute(0,2,1))
#         avg_x = self.mlp(self.avg_pool(query.permute(0,2,1)).permute(0,2,1))
#         attention_channel = self.sigmoid(max_x + avg_x)
#         query = query * attention_channel
#         return query
    
# class MLP(nn.Module):
#     """Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

## dynamic init query with text feature and density feature
class QueryLinguishDensityModule(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 定义可学习参数矩阵
        self.query_linguish_module = QueryLinguishModule(c)
        self.query_density_module = QueryAttentionModule(c)
    
    def forward(self, query, text_features, density_features, image_name=None, captions=None):
        query_with_linguish = self.query_linguish_module(query, text_features)
        query_with_linguish_density = self.query_density_module(query_with_linguish, density_features)
        # n = query_with_linguish_density.shape[0]
        # if image_name:
        #     import os
        #     # save_dir = os.path.mkdirs'query_vis/'
        #     vis_dir = 'query_vis/'
        #     name = image_name
        #     save_pth = os.path.join(vis_dir, name)
        #     os.makedirs(save_pth, exist_ok=True)
        #     # save_pth_caption = os.path.join(save_pth, f'{img_caps[b][1]}.jpg')
        #     plot_tsne_features_varying_radius(query[0:1].cpu(), 1, os.path.join(save_pth, 'query.jpg'), ['all'], perplexity=550, learning_rate=500, n_iter=1000, seed=314, grid_size=10, max_radis=4)
        #     for i in range(n):
        #         plot_tsne_features_varying_radius(query_with_linguish.cpu()[i:i+1], 1, os.path.join(save_pth, f'tsne_linguish_{i}.jpg'), captions[i:i+1], perplexity=550, learning_rate=500, n_iter=1000, seed=0, grid_size=10, color_setting=i, max_radis=5)
        #         plot_tsne_features_varying_radius(query_with_linguish_density.cpu()[i:i+1], 1, os.path.join(save_pth, f'tsne_linguish_density_{i}.jpg'),captions[i:i+1], perplexity=550, learning_rate=500, n_iter=1000, seed=0, grid_size=10, color_setting=i, max_radis=21)
        #     # 计算每一行的 L2 范数
        #     # norms = torch.norm(query_with_linguish_density, p=2, dim=2, keepdim=True)

        #     # # 对每一行进行归一化
        #     # normalized_features = query_with_linguish_density / norms
        #     # plot_tsne_features(query_with_linguish_density.cpu(), n, os.path.join(save_pth, 'tsne_linguish_density_query.jpg'),captions, perplexity=550, learning_rate=500, n_iter=1000)
        #     # plot_tsne_features(normalized_features.cpu(), n, os.path.join(save_pth, 'tsne_linguish_density_query_norm.jpg'),captions, perplexity=500, learning_rate=500, n_iter=1000, seed=314)
        #     plot_tsne_features_varying_radius(query_with_linguish.cpu(), n, os.path.join(save_pth, 'tsne_linguish_sum.jpg'),captions, perplexity=550, learning_rate=500, n_iter=1000, seed=0, grid_size=10)
        #     plot_tsne_features_varying_radius(query_with_linguish_density.cpu(), n, os.path.join(save_pth, 'tsne_linguish_density_sum.jpg'),captions, perplexity=550, learning_rate=500, n_iter=1000, seed=0, grid_size=10)
        #     plot_tsne_features(query_with_linguish_density.cpu(), n, os.path.join(save_pth, 'tsne_linguish_density_sum_xy.jpg'),captions, perplexity=550, learning_rate=500, n_iter=1000, seed=0)
        #     # plot_tsne_features(query_with_linguish_density.cpu(), 2, 'query_vis/tsne_linguish_density_query.jpg', perplexity=450, learning_rate=500, n_iter=500)
        #     #     plot_tsne_features(normalized_features.cpu(), n, os.path.join(save_pth, f'tsne_linguish_density_query_500_{100+i}.jpg'),captions, perplexity=500 , learning_rate=500, n_iter=1000, seed=100+i)
        
        return query_with_linguish_density
# from utils.util import plot_pca_features
# from utils.util import plot_tsne_features
# # plot_pca_features(query[0:1].cpu(), 1, 'query_vis/pca_query.jpg')
# # plot_pca_features(query_with_linguish.cpu(), 2, 'query_vis/pca_linguish_query.jpg')
# # plot_pca_features(query_with_linguish_density.cpu(), 2, 'query_vis/pca_linguish_density_query.jpg')

# plot_tsne_features(query[0:1].cpu(), 1, 'query_vis/tsne_query.jpg', perplexity=450, learning_rate=500, n_iter=500)
# plot_tsne_features(query_with_linguish.cpu(), 2, 'query_vis/tsne_linguish_query.jpg', perplexity=450, learning_rate=500, n_iter=500)
# plot_tsne_features(query_with_linguish_density.cpu(), 2, 'query_vis/tsne_linguish_density_query.jpg', perplexity=450, learning_rate=500, n_iter=500)
# for i in range(5): 
#     plot_tsne_features(query_with_linguish_density.cpu(), 2, f'query_vis/tsne_linguish_density_query_{i*10+400}.jpg', perplexity=i*10+400, learning_rate=500, n_iter=600)

# for i in range(16): 
#     plot_tsne_features(query_with_linguish_density.cpu(), 2, f'query_vis/tsne_linguish_density_query_450_500_{200+i*50}.jpg', perplexity=450, learning_rate=500, n_iter=250+i*50)