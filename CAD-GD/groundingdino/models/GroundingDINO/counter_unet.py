
"""
Counter modules.
"""
from torch import nn
import torch
import torch.nn.functional as F

# # 4 8 16 32 x
# class DensityRegressor(nn.Module):
#     def __init__(self, counter_dim):
#         super().__init__()
#         # 1/32 -> 1/16
#         self.conv0 = nn.Sequential(nn.Conv2d(counter_dim + 1, counter_dim, 7, padding=3),
#                                    nn.ReLU())
#         # 1/16 -> 1/8
#         self.conv1 = nn.Sequential(nn.Conv2d(counter_dim//4 + counter_dim + 1, counter_dim, 5, padding=2),
#                                    nn.ReLU())
#         # 1/8 -> 1/4
#         self.conv2 = nn.Sequential(nn.Conv2d(counter_dim//4 + counter_dim + 1, counter_dim, 3, padding=1),
#                                    nn.ReLU())
#         # 1/4 -> 1/2
#         self.conv3 = nn.Sequential(nn.Conv2d(counter_dim//4 + counter_dim + 1, counter_dim, 3, padding=1),
#                                    nn.ReLU())
#         # 1/2 -> 1
#         self.up2x = nn.Sequential(
#                                 nn.Conv2d(counter_dim, counter_dim//2, 1),
#                                 nn.ReLU())
#         # self.conv4 = nn.Sequential(
#         #                         nn.Conv2d(counter_dim//2, 1, 1),
#         #                         nn.ReLU())
#         self.conv4 = nn.Conv2d(counter_dim//8, 1, 1)
        
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self._weight_init_()
        
#     def forward(self, features, smaps, img_shape = [1000,1000], hidden_output=False):
#         x = torch.cat([features[3],smaps[3]], dim=1)
#         x1 = self.conv0(x)

#         # x = F.interpolate(x1, size = features[2].shape[-2:], mode='bilinear')
#         x = self.pixel_shuffle(x1) ## pixel shuffle upsample c//4
#         x = F.interpolate(x, size = features[2].shape[-2:], mode='bilinear')
#         x = torch.cat([x, features[2], smaps[2]], dim=1)
#         x2 = self.conv1(x)

#         # x = F.interpolate(x2, size = features[1].shape[-2:], mode='bilinear')
#         x = self.pixel_shuffle(x2)
#         x = F.interpolate(x, size = features[1].shape[-2:], mode='bilinear')
#         x = torch.cat([x, features[1], smaps[1]], dim=1)
#         x3 = self.conv2(x)

#         # x = F.interpolate(x3, size = features[0].shape[-2:], mode='bilinear')
#         x = self.pixel_shuffle(x3)
#         x = F.interpolate(x, size = features[0].shape[-2:], mode='bilinear')
#         x = torch.cat([x, features[0], smaps[0]], dim=1)
#         x4 = self.conv3(x)
#         x = self.up2x(x4)
        
#         x = self.pixel_shuffle(x)
        
#         x = F.interpolate(x, size = img_shape, mode='bilinear')
#         x = self.conv4(x)

#         if hidden_output:
#             return x, [x4,x3,x2,x1]
#         else:
#             return x

#     def _weight_init_(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 # nn.init.kaiming_uniform_(
#                 #         m.weight, 
#                 #         mode='fan_in', 
#                 #         nonlinearity='relu'
#                 #         )
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# 4 8 16 32 x
class DensityRegressor(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        # 1/32 -> 1/16
        self.conv0 = nn.Sequential(nn.Conv2d(counter_dim + 1, counter_dim, 7, padding=3),
                                   nn.ReLU())
        # 1/16 -> 1/8
        self.conv1 = nn.Sequential(nn.Conv2d(counter_dim * 2 + 1, counter_dim, 5, padding=2),
                                   nn.ReLU())
        # 1/8 -> 1/4
        self.conv2 = nn.Sequential(nn.Conv2d(counter_dim * 2 + 1, counter_dim, 3, padding=1),
                                   nn.ReLU())
        # 1/4 -> 1/2
        self.conv3 = nn.Sequential(nn.Conv2d(counter_dim * 2 + 1, counter_dim, 3, padding=1),
                                   nn.ReLU())
        # 1/2 -> 1
        self.up2x = nn.Sequential(
                                nn.Conv2d(counter_dim, counter_dim//2, 1),
                                nn.ReLU())
        # self.conv4 = nn.Sequential(
        #                         nn.Conv2d(counter_dim//2, 1, 1),
        #                         nn.ReLU())
        self.conv4 = nn.Conv2d(counter_dim//2, 1, 1)
        # self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self._weight_init_()
        
    def forward(self, features, smaps, img_shape = [1000,1000], hidden_output=False):
        x = torch.cat([features[3],smaps[3]], dim=1)
        x1 = self.conv0(x)

        x = F.interpolate(x1, size = features[2].shape[-2:], mode='bilinear')
        # x = self.pixel_shuffle(x1) ##
        x = torch.cat([x, features[2], smaps[2]], dim=1)
        x2 = self.conv1(x)

        x = F.interpolate(x2, size = features[1].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[1], smaps[1]], dim=1)
        x3 = self.conv2(x)

        x = F.interpolate(x3, size = features[0].shape[-2:], mode='bilinear')
        x = torch.cat([x, features[0], smaps[0]], dim=1)
        x4 = self.conv3(x)
        x = self.up2x(x4)

        x = F.interpolate(x, size = img_shape, mode='bilinear')
        x = self.conv4(x)

        if hidden_output:
            return x, [x4,x3,x2,x1]
        else:
            return x

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    img_shape = [400, 600]
    encode_vision_feature = torch.rand(2,19947,256)
    spatial_shapes = torch.tensor([[100, 150],
        [ 50,  75],
        [ 25,  38],
        [ 13,  19]])
    N_, S_, C_ = encode_vision_feature.shape
    _cur = 0
    cnn = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # mask_flatten_ = mask_flatten[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
        memory_flatten_ = encode_vision_feature[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, -1)
        cnn.append(memory_flatten_.permute(0,3,1,2))
        _cur += H_ * W_
    model = DensityRegressor(counter_dim=256)
    output = model(cnn, img_shape)
    print(1)
