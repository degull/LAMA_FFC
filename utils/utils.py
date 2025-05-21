# utils.py
import os
from torchvision.utils import save_image

def save_sample_images(fake, real, mask, save_dir, epoch):
    out = fake.detach().cpu()
    gt = real.detach().cpu()
    msk = mask.detach().cpu()
    stacked = (out * (1 - msk) + msk * gt)

    save_image(stacked, os.path.join(save_dir, f"epoch{epoch}_stacked.png"), nrow=4)
    save_image(out, os.path.join(save_dir, f"epoch{epoch}_output.png"), nrow=4)
    save_image(gt, os.path.join(save_dir, f"epoch{epoch}_gt.png"), nrow=4)
    save_image(msk, os.path.join(save_dir, f"epoch{epoch}_mask.png"), nrow=4)
