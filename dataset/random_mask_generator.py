# utils/random_mask_generator.py
import numpy as np
import cv2
import random
import torch

def generate_random_mask(image_shape, max_num_rectangles=5):
    """
    이미지와 동일한 크기의 랜덤 마스크를 생성합니다.
    사각형 블록 형태의 마스크를 N개 생성.
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(random.randint(1, max_num_rectangles)):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(x1 + 1, width)
        y2 = random.randint(y1 + 1, height)
        mask[y1:y2, x1:x2] = 255

    # 마스크 확장
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # torch tensor로 변환
    mask = torch.from_numpy(mask).float().div(255.0).unsqueeze(0)  # (1, H, W)
    return mask
