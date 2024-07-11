import torch
import torch.nn as nn
from torch.nn.functional import linear, pad, softmax, dropout # containing original multi_head_attention_forward implementation
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.nn.modules.activation import Parameter # containing original MultiheadAttention implementation
# from torch.nn.modules.linear import _LinearWithBias
import torch.nn.functional as F

import numpy as np
import torchvision
from .box_ops import box_cxcywh_to_xyxy

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)

# 使用XavierFill初始化的FC
def make_fc(dim_in, hidden_dim, a=1):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
        a: negative slope
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=a)
    nn.init.constant_(fc.bias, 0)
    return fc

class RelationFeatureExtractor(nn.Module):
    def __init__(self, args, in_channels, out_dim, num_objs):
        super(RelationFeatureExtractor, self).__init__()
        self.args = args

        sub_dim = self.args.hidden_dim
        obj_dim = self.args.hidden_dim

        # spatial relation
        if self.args.use_spatial_relation:
            semantic_dim = 300
            self.relation_embedding = nn.Embedding(5, semantic_dim)
            sub_dim += semantic_dim
            obj_dim += semantic_dim

        self.sub_fc = nn.Sequential(
            make_fc(sub_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )

        self.obj_fc = nn.Sequential(
            make_fc(obj_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )


    def forward(self, head_boxes, tail_boxes, head_feats, tail_feats, obj_label_logits=None):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (cx_cy_wh, nomalized to 0-1)
            rel_pairs: Nx2
        """

        # head & tail features
        # relation_feats = (head_feats + tail_feats) / 2.0
        head_boxes = box_cxcywh_to_xyxy(head_boxes).clamp(0, 1)
        tail_boxes = box_cxcywh_to_xyxy(tail_boxes).clamp(0, 1)

        if self.args.use_spatial_relation:
            spatial_relation = self.generate_spatial_relation(head_boxes, tail_boxes)
            spatial_relation = torch.zeros((spatial_relation.size(0), spatial_relation.size(1), 5)).to(spatial_relation.device).scatter_(2, spatial_relation.unsqueeze(-1), 1.0)
            spatial_relation = spatial_relation @ self.relation_embedding.weight
            head_feats = torch.cat([head_feats, spatial_relation], dim=-1)
            tail_feats = torch.cat([tail_feats, spatial_relation], dim=-1)

        head_feats = self.sub_fc(head_feats)
        tail_feats = self.obj_fc(tail_feats)

        return head_feats, tail_feats

    # cxcywh
    def generate_spatial_relation(self, head_boxes, tail_boxes):
        alpha = self.args.spatial_alpha
        above = head_boxes[:, :, 1] > (tail_boxes[:, :, 1] + alpha * tail_boxes[:, :, 3])
        below = head_boxes[:, :, 1] < (tail_boxes[:, :, 1] - alpha * tail_boxes[:, :, 3])

        around = ((tail_boxes[:, :, 1] - alpha * tail_boxes[:, :, 3]) < head_boxes[:, :, 1]) * (head_boxes[:, :, 1] < (tail_boxes[:, :, 1] + alpha * tail_boxes[:, :, 3])) *\
                ((head_boxes[:, :, 0] < (tail_boxes[:, :, 0] - alpha * tail_boxes[:, :, 2])) + (head_boxes[:, :, 0] > (tail_boxes[:, :, 0] + alpha * tail_boxes[:, :, 2])))
        around = around > 0

        within = ((tail_boxes[:, :, 1] - alpha * tail_boxes[:, :, 3]) < head_boxes[:, :, 1]) * (head_boxes[:, :, 1] < (tail_boxes[:, :, 1] + alpha * tail_boxes[:, :, 3])) *\
                ((head_boxes[:, :, 0] > (tail_boxes[:, :, 0] - alpha * tail_boxes[:, :, 2])) * (head_boxes[:, :, 0] < (tail_boxes[:, :, 0] + alpha * tail_boxes[:, :, 2])))
        within = within > 0

        contain = (tail_boxes[:, :, 2] * tail_boxes[:, :, 3]) < 1e-5

        relation = above * 1 + below * 2 + around * 3 + within * 4

        relation = relation * (1 - contain*1)

        return relation






