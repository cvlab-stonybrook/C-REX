import copy
from typing import List
import json
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
import torch.utils.checkpoint as checkpoint
from .transformer_vanilla import TransformerEncoderLayer
from .fuse_modules import BiAttentionBlock

from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)

## deformable attention layer for vision feature
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0: 
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None: 
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share 
                ) 
            if feature_fusion_layer is not None: 
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                ) 
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint 
        self.use_transformer_ckpt = use_transformer_ckpt 

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        if self.text_layers: 
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text, device=memory_text.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(
                    position_ids[..., None], num_pos_feats=256, exchange_xy=False
                )

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask,
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](
                        v=output,
                        l=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text

class CADM(nn.Module):
    def __init__(self,use_text_enhancer, d_model,text_nhead,text_dim_feedforward,text_dropout,
                 fusion_dropout, fusion_droppath,
                 dim_feedforward, use_vision_selection, dropout, activation, num_feature_levels, nhead, enc_n_points):
        super().__init__()
        # setting the context aware selection module
        self.use_text_enhancer = use_text_enhancer
        self.use_vision_selection = use_vision_selection

        if self.use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=text_nhead,
                    dim_feedforward=text_dim_feedforward,
                    dropout=text_dropout,
                )
        else:
            text_enhance_layers = None

        context_selection_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=text_dim_feedforward,
                num_heads=text_nhead,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        
        if self.use_vision_selection:
            vision_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        else:
            vision_layer = None

        num_encoder_layers = 1
        self.decoder = TransformerEncoder(
            vision_layer,
            num_encoder_layers,
            d_model=d_model,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=context_selection_layer,
            use_checkpoint=True, # True
            use_transformer_ckpt=True, # True
        )

    def forward(self,
        # for images
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        # for texts
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,):

        memory, memory_text = self.decoder( 
            src,
            pos=pos,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=key_padding_mask,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=position_ids,
            text_self_attention_masks=text_self_attention_masks,
        )

        return memory, memory_text

if __name__ == '__main__':
    # model = MLCCM(in_channels=256,hidden_dim=512,out_dim=512)
    encode_vision_feature = torch.rand(2,19947,256)
    lvl_pos_embed_flatten = torch.rand(2,19947,256)
    level_start_index = torch.tensor([0,25000,18750,19700])
    spatial_shapes = torch.tensor([[100, 150],
        [ 50,  75],
        [ 25,  38],
        [ 13,  19]])
    valid_ratios = torch.tensor([[[1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.],
         [1., 1.],
         [1., 1.]]])
    mask_flatten = torch.zeros([2,19947], dtype=torch.bool)
    memory_text = torch.rand(2,6,256)
    text_attention_mask = torch.zeros([2,6], dtype=torch.bool)
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0],
        [0, 0, 1, 2, 0, 0]])
    attention_mask = torch.tensor([[[ True, False, False, False, False, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False,  True,  True,  True,  True, False],
         [False, False, False, False, False,  True]],

        [[ True, False, False, False, False, False],
         [False,  True,  True,  True, False, False],
         [False,  True,  True,  True, False, False],
         [False,  True,  True,  True, False, False],
         [False, False, False, False,  True, False],
         [False, False, False, False, False,  True]]])
    
    model = CADM(use_text_enhancer=True, d_model=256 ,text_nhead=4, text_dim_feedforward=1024,text_dropout=0.,
                 fusion_dropout=0., fusion_droppath=0.1,
                 use_vision_selection=True, dim_feedforward=2048, dropout=0., activation='relu', num_feature_levels=4, nhead=8 ,enc_n_points=4)
    
    memory, memory_text = model(encode_vision_feature,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=position_ids,
            text_self_attention_masks=attention_mask)
    print(1)
    # score = model(x)
    # print(1)
    # density_feature = torch.rand(8,512,256,256)
    # model = CACCM(256,512,512)
    # output = model(x1, text)
    # # model = CCM(256,512,512)
    # # model = CGFE(x1, text)
    # x_enhanced = model(x, density_feature)
    # print(1)


    # density_feature, score = model(x1)
    # label = torch.tensor([0,1,2,3,0,1,2,3])
    # cri = nn.CrossEntropyLoss()
    # loss = cri(score, label)
    # pred_index = torch.argmax(score,dim=1)
    # accuracy = torch.sum(pred_index == label)

    # print(score)
    
