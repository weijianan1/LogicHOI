
import torch.nn.functional as F
import json
import numpy as np
# from .ltn_utils import *
import math
import torch
import random
from datasets.vcoco_text_label import *

def obj2hoi_logic(hois_logits, obj2hoi, args):
    loss_me_obj2hoi = loss_mutual_exclusion_flat(hois_logits.float(), obj2hoi.float(), p=args.me_p)
    return {"loss_obj2hoi_me": loss_me_obj2hoi}

def verb2hoi_logic(hois_logits, verb2hoi, args):
    loss_me_verb2hoi = loss_mutual_exclusion_flat(hois_logits.float(), verb2hoi.float(), p=args.me_p)
    return {"loss_verb2hoi_me": loss_me_verb2hoi}

def loss_mutual_exclusion_flat(hois, obj2hoi, p=2):
    predicate_logits = 1 - hois.unsqueeze(-1) * hois.unsqueeze(-2)  # b,nh,nh
    mask = 1 - ((obj2hoi.T @ obj2hoi) > 0) * 1.0  # nh,nh
    mask = mask.unsqueeze(0).repeat(hois.size(0), 1, 1)
    sat_agg = AggPowerMeanError(predicate_logits, mask, p)

    return 1 - sat_agg

def AggPowerMeanError(predicate_hois, mask, p):
    #AggPowerMeanError
    logits = 1. - predicate_hois + 1e-9
    logits = torch.pow(logits, p)
    logits[mask==0.] = 0
    return 1 - torch.pow(logits.sum(-1)/(mask.sum(-1)+1e-3), exponent=(1./p)).mean()






