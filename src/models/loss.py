import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction") or (task == "dynamic_imputation"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        if config.get('num_classes', 2) == 2:  # Binary classification
            pos_weight = torch.tensor([config.get('pos_weight', 4.0)])
            return WeightedBCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:  # Multi-class
            return MulticlassClassificationLoss(pos_weight=config.get('pos_weight'))

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample

    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of all model parameters"""
    device = next(model.parameters()).device
    l2_loss = torch.tensor(0.0, device=device)
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2) ** 2
    return l2_loss


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='none'):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight)
        print(f"Using pos_weight: {pos_weight}")
        self.reduction = reduction

    def forward(self, inp, target):
        # Convert target to float and move to same device as input
        target = target.float().to(inp.device)
        
        # Handle single-label case by converting to one-hot
        if target.dim() == 1 or target.size(1) == 1:
            target = F.one_hot(target.long().squeeze(), num_classes=2).float().to(inp.device)
            
        # Ensure pos_weight is on correct device
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(inp.device)
            
        return F.binary_cross_entropy_with_logits(
            inp, target,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class MulticlassClassificationLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inp, target):
        # Convert target to one-hot if it's not already
        if target.dim() == 1 or target.size(1) == 1:
            target = torch.zeros_like(inp).scatter_(1, target.long().view(-1, 1), 1)
        return self.loss_fn(inp, target.float())
