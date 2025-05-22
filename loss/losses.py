# loss/losses.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, device, resize=True):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg_layers = vgg16.to(device)
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.resize = resize
        self.device = device

    def forward(self, input, target):
        def normalize_batch(batch):
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
            return (batch - mean) / std

        input = normalize_batch(input)
        target = normalize_batch(target)

        if self.resize:
            input = F.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        features_input = self.vgg_layers(input)
        features_target = self.vgg_layers(target)

        return F.l1_loss(features_input, features_target)


class GANLoss(nn.Module):
    def __init__(self, gan_mode="hinge"):
        super().__init__()
        self.gan_mode = gan_mode

    def forward(self, pred, target_is_real):
        if self.gan_mode == "hinge":
            if target_is_real:
                return torch.mean(F.relu(1.0 - pred))
            else:
                return torch.mean(F.relu(1.0 + pred))
        else:
            raise NotImplementedError(f"Unsupported GAN mode: {self.gan_mode}")


class MultiScaleGANLoss(nn.Module):
    def __init__(self, gan_mode="hinge"):
        super().__init__()
        self.gan_loss = GANLoss(gan_mode)

    def forward(self, preds, target_is_real):
        return sum(self.gan_loss(pred, target_is_real) for pred in preds) / len(preds)


class LaMaLoss(nn.Module):
    def __init__(self, device, lambda_l1=1.0, lambda_perc=0.1, lambda_gan=0.01):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(device)
        self.gan = MultiScaleGANLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_gan = lambda_gan

    def forward(self, fake_img, real_img, pred_fake, pred_real=None):
        loss_l1 = self.l1(fake_img, real_img)
        loss_perc = self.perceptual(fake_img, real_img)
        loss_gan = self.gan(pred_fake, True)

        if pred_real is not None:
            loss_d_real = self.gan(pred_real, True)
            loss_d_fake = self.gan(pred_fake, False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
        else:
            loss_d = torch.tensor(0.0, device=fake_img.device)

        loss_g_total = (
            self.lambda_l1 * loss_l1
            + self.lambda_perc * loss_perc
            + self.lambda_gan * loss_gan
        )

        return {
            "loss_l1": loss_l1,
            "loss_perceptual": loss_perc,
            "loss_gan": loss_gan,
            "loss_generator_total": loss_g_total,
            "loss_discriminator": loss_d,
        }
