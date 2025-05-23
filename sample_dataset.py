import os
import random
import glob
import shutil

src_dir = r"E:\이미지개선\FFT\data\Places365\train"
dst_dir = r"E:\이미지개선\FFT\data\Places365\sampled_train"
num_samples = 100000

# 1. 모든 이미지 파일 경로 수집 (하위 폴더 포함)
image_paths = glob.glob(os.path.join(src_dir, "*", "*.jpg"))

print(f"전체 이미지 수: {len(image_paths)}")

# 2. 무작위로 10만장 샘플링
sampled_paths = random.sample(image_paths, num_samples)

# 3. 저장 디렉토리 생성
os.makedirs(dst_dir, exist_ok=True)

# 4. 샘플링된 이미지 복사 (파일명 중복 방지 위해 폴더명_파일명 구조로 저장)
for path in sampled_paths:
    class_name = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    new_name = f"{class_name}_{filename}"
    shutil.copy(path, os.path.join(dst_dir, new_name))

print(f"✅ {num_samples}장 샘플링 완료 → {dst_dir}")



# sampling
""" import os
import random
import glob

# 설정
src_dir = r"E:\이미지개선\FFT\data\Places365\train"     # 전체 이미지가 들어있는 상위 디렉토리
output_txt = "sampled_image_paths.txt"                 # 저장할 텍스트 파일
num_samples = 100000                                   # 샘플링할 이미지 수

# 1. 전체 이미지 경로 수집 (하위 폴더까지)
image_paths = glob.glob(os.path.join(src_dir, "*", "*.jpg"))
print(f"전체 이미지 수: {len(image_paths)}")

# 2. 샘플링
sampled_paths = random.sample(image_paths, num_samples)

# 3. 텍스트 파일로 경로 저장
with open(output_txt, "w") as f:
    for path in sampled_paths:
        f.write(path + "\n")

print(f"✅ 경로 샘플링 완료: {num_samples}개 경로 → {output_txt}")
 """