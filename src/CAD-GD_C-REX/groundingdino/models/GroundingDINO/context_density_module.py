import copy
from typing import List
import json
import torch
import torch.nn.functional as F
from torch import nn


## TODO: Simple Context Guide Module
### object: use the cosine similarity as the guidance feature
class SCGM(nn.Module):
    def __init__(self, mode='01'):
        super().__init__()
        self.mode = mode

    # vision_feature: b c h w, text_feature: b n c 
    def forward(self, vision_feature, text_feature):
        b,c,h,w = vision_feature.shape
        vision_feature_flatten = vision_feature.view(b,c,-1)
        context_similarity = torch.matmul(text_feature, vision_feature_flatten)
        context_similarity = context_similarity.mean(1)
        max_sim = torch.max(context_similarity,dim=1)[0].unsqueeze(-1)
        min_sim = torch.min(context_similarity,dim=1)[0].unsqueeze(-1)
        context_similarity = (context_similarity - min_sim) / (max_sim - min_sim)
        if self.mode == '01':
            context_similarity = context_similarity.view(b, h, w)
            return context_similarity
        elif self.mode == '-11':
            context_similarity = context_similarity * 2 - 1
            context_similarity = context_similarity.view(b, h, w)
            return context_similarity

## TODO: Multi-Head Context Guide Module
### object: use the cross-attention multi-head design as the multi-head feature guide

# context density aware grounding dino
# context-aware categorical counting module:
class CACCM(nn.Module):
    """This is the reproduce of Categorical Counting Module"""
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 out_dim,
                 guide_mode = 'cat',
                 norm_mode ='01'):
        super().__init__()
        self.guide_mode = guide_mode
        if guide_mode == 'cat':
            self.DE = nn.Sequential(
                nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=1,stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
                nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
                nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                # nn.GroupNorm(32, hidden_dim),
            )
        elif guide_mode == 'mul':
            self.DE = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1,stride=1,padding=0),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
                nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
                nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
                nn.ReLU(),
                # nn.GroupNorm(32, hidden_dim),
            )
        self.sim_module = SCGM(mode=norm_mode)
        self.ClsHead = ClassificationHead(dim=out_dim, num_classes=4)

    def forward(self, x, t):
        similarity_map = self.sim_module(x,t).unsqueeze(1)
        if self.guide_mode == 'cat':
            x = torch.cat([similarity_map, x], dim=1)
        elif self.guide_mode == 'mul':
            x = x + x * similarity_map
        density_feature = self.DE(x)
        score = self.ClsHead(density_feature)
        return density_feature, score

# context density aware grounding dino
# multi-layer categorical counting module:
class MLCCM(nn.Module):
    """This is the reproduce of Categorical Counting Module"""
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 out_dim):
        super().__init__()

        self.DE0 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            # nn.GroupNorm(32, hidden_dim),
        )

        self.DE1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            # nn.GroupNorm(32, hidden_dim),
        )

        self.DE2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            # nn.GroupNorm(32, hidden_dim),
        )

        self.DE3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            # nn.GroupNorm(32, hidden_dim),
        )

        self.DE = nn.ModuleList([self.DE0, self.DE1, self.DE2, self.DE3])

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层
        self.flatten = nn.Flatten()  # 扁平化层
        self.fc = nn.Linear(out_dim * 4, 4)  # 全连接层

    def forward(self, x):
        density_features = []
        for i in range(4):
            density_feature = self.DE[i](x[i])
            density_feature = self.global_avg_pool(density_feature)
            density_feature = self.flatten(density_feature)
            density_features.append(density_feature)
        density_feature = torch.cat(density_features,dim=1)
        score = self.fc(density_feature)
        return density_feature, score

# context density aware grounding dino
# categorical counting module:
class CCM(nn.Module):
    """This is the reproduce of Categorical Counting Module"""
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 out_dim):
        super().__init__()

        self.DE = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, dilation=1, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=2, kernel_size=3, stride=1, padding=2),
            nn.Conv2d(hidden_dim, out_dim, dilation=3, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            # nn.GroupNorm(32, hidden_dim),
        )
        self.ClsHead = ClassificationHead(dim=out_dim, num_classes=4)

    def forward(self, x):
        density_feature = self.DE(x)
        score = self.ClsHead(density_feature)
        return density_feature, score

class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层
        self.flatten = nn.Flatten()  # 扁平化层
        self.fc = nn.Linear(dim, num_classes)  # 全连接层
        # self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.global_avg_pool(x)  
        x = self.flatten(x)  
        x = self.fc(x)
        return x

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CGFE(nn.Module):
    """This is the reproduce of Counting-Guided Feature Enhancement"""
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 ):
        super().__init__()

        ## Spatial cross-attention map
        ### down multiple layer feature
        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=4,stride=4,ceil_mode=True),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=8,stride=8,ceil_mode=True),
        )

        self.down = nn.ModuleList([self.down0, self.down1, self.down2, self.down3])

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        ### get the attention

        self.spatial_attention = nn.ModuleList([nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) for i in range(4)])

        ## Channel attention map
        self.max_pool2d = nn.AdaptiveMaxPool2d((1,1))
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1,1))

        self.mlp = MLP(256, 256, 256, num_layers=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vision_features, density_feature):
        # spatial weight forward
        # attentions_spatial = []
        for i, down_layer in enumerate(self.down):
            x = down_layer(density_feature)
            b, c, h, w = x.shape
            max_x = self.max_pool(x.permute(0,2,3,1).view(b, -1, c)).view(b, h, w, -1)
            avg_x = self.avg_pool(x.permute(0,2,3,1).view(b, -1, c)).view(b, h, w, -1)
            cat_x = torch.cat([max_x,avg_x],dim=-1)
            cat_x = cat_x.permute(0,3,1,2)
            attention_spatial = self.sigmoid(self.spatial_attention[i](cat_x))
            # attentions_spatial.append(attention_spatial)
            vision_features[i] = vision_features[i] * attention_spatial
        # channel weight forward
        for i, vision_feature in enumerate(vision_features):
            max_x = self.mlp(self.max_pool2d(vision_feature).squeeze())
            avg_x = self.mlp(self.avg_pool2d(vision_feature).squeeze())
            attention_channel = self.sigmoid(max_x + avg_x)
            vision_features[i] = attention_channel.unsqueeze(-1).unsqueeze(-1) * vision_features[i]
        return vision_features
    
if __name__ == '__main__':
    model = MLCCM(in_channels=256,hidden_dim=512,out_dim=512)
    x1 = torch.rand(8,256,256,256)
    x2 = torch.rand(8,256,128,128)
    x3 = torch.rand(8,256,64,64)
    x4 = torch.rand(8,256,32,32)
    x = [x1,x2,x3,x4]
    text = torch.rand(8,16,256)
    score = model(x)
    print(1)
    density_feature = torch.rand(8,512,256,256)
    model = CACCM(256,512,512)
    output = model(x1, text)
    # model = CCM(256,512,512)
    # model = CGFE(x1, text)
    x_enhanced = model(x, density_feature)
    print(1)


    # density_feature, score = model(x1)
    # label = torch.tensor([0,1,2,3,0,1,2,3])
    # cri = nn.CrossEntropyLoss()
    # loss = cri(score, label)
    # pred_index = torch.argmax(score,dim=1)
    # accuracy = torch.sum(pred_index == label)

    # print(score)
    
