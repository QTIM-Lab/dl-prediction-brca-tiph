# PyTorch and PyTorch Toolbelt Imports
import torch.nn as nn
from pytorch_toolbelt import losses as L



# Function: create loss
def create_loss(conf_loss, w1=1.0, w2=0.5):

    loss = None
    
    if hasattr(nn, conf_loss): 
        loss = getattr(nn, conf_loss)() 
    elif conf_loss == "focal":
        loss = L.BinaryFocalLoss()
    elif conf_loss == "jaccard":
        loss = L.BinaryJaccardLoss()
    elif conf_loss == "jaccard_log":
        loss = L.BinaryJaccardLoss()
    elif conf_loss == "dice":
        loss = L.BinaryDiceLoss()
    elif conf_loss == "dice_log":
        loss = L.BinaryDiceLogLoss()
    elif conf_loss == "dice_log":
        loss = L.BinaryDiceLogLoss()
    elif conf_loss == "bce+lovasz":
        loss = L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryLovaszLoss(), w1, w2)
    elif conf_loss == "lovasz":
        loss = L.BinaryLovaszLoss()
    elif conf_loss == "bce+jaccard":
        loss = L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryJaccardLoss(), w1, w2)
    elif conf_loss == "bce+log_jaccard":
        loss = L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryJaccardLogLoss(), w1, w2)
    elif conf_loss == "bce+log_dice":
        loss = L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryDiceLogLoss(), w1, w2)
    elif conf_loss == "reduced_focal":
        loss = L.BinaryFocalLoss(reduced=True)
    else:
        assert False, "Invalid loss."

    return loss
