# test.py
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.generator import LaMaGenerator
from dataset.lama_dataset import LaMaDataset

def test():
    # 🔧 설정
    image_dir = "data/val"                     # ✅ 테스트 이미지 디렉토리
    save_dir = "results"                       # ✅ 결과 저장 폴더
    checkpoint = "checkpoints/lama_epoch20.pth"  # ✅ 학습된 LaMa Generator 체크포인트
    image_size = 256
    mask_type = "irregular"  # or "box"

    # 🔧 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 데이터셋 & 로더
    dataset = LaMaDataset(
        image_dir=image_dir,
        mask_type=mask_type,
        image_size=image_size,
        phase="val"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ✅ 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 모델 초기화 및 로드
    model = LaMaGenerator(in_channels=4).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # ✅ 테스트 루프
    for i, batch in enumerate(dataloader):
        input_tensor = batch["input"].to(device)     # [masked_img + mask] (4ch)
        gt_image = batch["image"].to(device)         # 원본
        mask = batch["mask"].to(device)              # 마스크

        with torch.no_grad():
            output = model(input_tensor)

        # 🔽 결과 저장
        save_image(output, os.path.join(save_dir, f"{i:03}_output.png"))
        save_image(gt_image, os.path.join(save_dir, f"{i:03}_gt.png"))
        save_image(mask, os.path.join(save_dir, f"{i:03}_mask.png"))

        # 마스킹된 입력 이미지도 저장 (for comparison)
        masked_img = input_tensor[:, :3, :, :]  # 앞 3채널이 masked image
        save_image(masked_img, os.path.join(save_dir, f"{i:03}_masked.png"))

if __name__ == "__main__":
    test()
