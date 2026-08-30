import torch

from swift_plugin.segmentation import SEG_TOKEN, dice_loss


def test_seg_token_constant():
    assert SEG_TOKEN == "[SEG]"


def test_dice_loss_is_zero_for_confident_match():
    logits = torch.tensor([[[20.0, -20.0], [-20.0, 20.0]]])
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert dice_loss(logits, targets).item() < 1e-6
