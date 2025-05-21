# loss/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor

# ✅ L1 Reconstruction Loss
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


# ✅ Adversarial Loss (PatchGAN or Multi-scale)
class AdversarialLoss(nn.Module):
    def __init__(self, type="hinge"):
        super().__init__()
        assert type in ["hinge", "bce"]
        self.type = type

    def forward(self, preds, is_real):
        def loss_single(pred):
            if self.type == "hinge":
                return torch.mean(F.relu(1.0 - pred)) if is_real else torch.mean(F.relu(1.0 + pred))
            else:
                labels = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
                return F.binary_cross_entropy_with_logits(pred, labels)

        # ✅ 리스트 형태 (multi-scale)인 경우
        if isinstance(preds, list):
            return sum([loss_single(p) for p in preds]) / len(preds)
        else:
            return loss_single(preds)


# ✅ Feature Matching Loss (중간 feature 비교)
class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, real_feats, fake_feats):
        if isinstance(real_feats, list) and isinstance(fake_feats, list):
            loss = 0.0
            for rf, ff in zip(real_feats, fake_feats):
                loss += self.loss(ff, rf.detach())
            return loss / len(real_feats)
        else:
            return self.loss(fake_feats, real_feats.detach())


# ✅ VGG 기반 Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1"), weight=1.0):
        super().__init__()
        self.weight = weight
        vgg = vgg19(pretrained=True).features
        self.vgg = create_feature_extractor(vgg, return_nodes={l: l for l in layers})
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_feats = self.vgg(pred)
        target_feats = self.vgg(target)
        loss = 0.0
        for k in pred_feats.keys():
            loss += self.loss(pred_feats[k], target_feats[k].detach())
        return self.weight * loss


# ✅ 전체 종합 Loss
class LaMaLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, adv_weight=0.1, fm_weight=0.1):
        super().__init__()
        self.rec_loss = ReconstructionLoss()
        self.percep_loss = PerceptualLoss(weight=perceptual_weight)
        self.adv_loss = AdversarialLoss(type="hinge")
        self.fm_loss = FeatureMatchingLoss()
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight

    def forward(self, pred, target, disc_pred_fake, disc_feats_real=None, disc_feats_fake=None):
        loss_rec = self.rec_loss(pred, target)
        loss_percep = self.percep_loss(pred, target)
        loss_adv = self.adv_loss(disc_pred_fake, True)

        if disc_feats_real is not None and disc_feats_fake is not None:
            loss_fm = self.fm_loss(disc_feats_real, disc_feats_fake)
        else:
            loss_fm = 0.0

        loss_total = (
            loss_rec +
            loss_percep +
            self.adv_weight * loss_adv +
            self.fm_weight * loss_fm
        )

        return {
            "loss_total": loss_total,
            "loss_rec": loss_rec,
            "loss_percep": loss_percep,
            "loss_adv": loss_adv,
            "loss_fm": loss_fm
        }


"""
Adversarial Loss (PatchGAN 기반)

Feature Matching Loss

Perceptual Loss (VGG-19 기반)

L1 Reconstruction Loss
"""