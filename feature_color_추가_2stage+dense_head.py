import shutil
import zipfile

# 1. Drive에서 로컬로 복사 (빠름)
shutil.copy("/content/drive/MyDrive/open.zip", "/content/open.zip")

# 2. 압축 풀기
with zipfile.ZipFile("/content/open.zip", "r") as zip_ref:
    zip_ref.extractall("/content/")

# 3. 확인
!ls /content/train

import os
import glob
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# train 경로
train_dir = '/content/train'

# test 경로
test_dir = '/content/test'

# sample_submission (추론 때 사용)
sample_submission_path = '/content/sample_submission.csv'

import torch
import gc

def show_memory_status():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB 단위
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB 단위
    print(f"📊 현재 GPU 메모리 상태: Allocated = {allocated:.2f} MB | Reserved = {reserved:.2f} MB")

# 현재 CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    print("🔍 초기화 전 GPU 메모리 상태:")
    show_memory_status()

    # GPU 캐시 및 메모리 초기화
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    print("\n GPU 메모리 초기화 완료")
    print(" 초기화 후 GPU 메모리 상태:")
    show_memory_status()
else:
    print("CUDA 사용 불가")

import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import os

class CarImageDataset(Dataset):
    def __init__(self, file_list, class_to_idx, transform=None, use_aspect=False, use_color=False):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_aspect = use_aspect
        self.use_color = use_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        # Step 1: Load Image (RGB 고정)
        image_pil = Image.open(path).convert("RGB")

        # Step 2: Calculate Features if needed
        width, height = image_pil.size
        aspect_ratio = np.array([width / height], dtype=np.float32)

        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        color_mean = image_cv2.mean(axis=(0, 1))
        color_mean = color_mean[::-1]
        color_mean = np.array(color_mean / 255.0, dtype=np.float32)

        # Step 3: Apply Transform
        if self.transform:
            image = self.transform(image_pil)
        else:
            image = transforms.ToTensor()(image_pil)  # fallback

        #  Step 4: Extract label
        class_name = os.path.basename(os.path.dirname(path))
        label = self.class_to_idx[class_name]

        #  Step 5: Return according to mode
        if self.use_aspect and self.use_color:
            return image, torch.tensor(aspect_ratio), torch.tensor(color_mean), label
        elif self.use_aspect:
            return image, torch.tensor(aspect_ratio), label
        elif self.use_color:
            return image, torch.tensor(color_mean), label
        else:
            return image, label

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

#  전체 JPG 파일 불러오기 (Train)
file_list = glob.glob('/content/train/*/*.jpg')

#  클래스명 추출 (폴더명)
def extract_class_name_jpg(path):
    return os.path.basename(os.path.dirname(path))

class_names = sorted(set(extract_class_name_jpg(f) for f in file_list))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

print(f" 클래스 수: {len(class_to_idx)}")  # 396개 나와야 정상

#  라벨 생성
labels = [class_to_idx[extract_class_name_jpg(f)] for f in file_list]

#  Train/Val Split
from sklearn.model_selection import train_test_split

train_files, val_files = train_test_split(file_list, test_size=0.1, stratify=labels, random_state=42)

#  Transform 정의
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 확장형 Dataset 클래스 (앞서 만든 버전 사용!)
class CarImageDataset(Dataset):
    def __init__(self, file_list, class_to_idx, transform=None, use_aspect=False, use_color=False):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_aspect = use_aspect
        self.use_color = use_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        #  Load Image (RGB 고정)
        image_pil = Image.open(path).convert("RGB")

        #  Feature: Aspect Ratio
        width, height = image_pil.size
        aspect_ratio = torch.tensor([width / height], dtype=torch.float32)

        #  Feature: Dominant Color (mean RGB)
        image_np = np.array(image_pil)
        color_mean = image_np.mean(axis=(0, 1)) / 255.0  # Normalize to 0~1
        color_mean = torch.tensor(color_mean, dtype=torch.float32)

        #  Transform
        if self.transform:
            image = self.transform(image_pil)
        else:
            image = transforms.ToTensor()(image_pil)

        #  Label
        class_name = extract_class_name_jpg(path)
        label = self.class_to_idx[class_name]

        #  Return mode
        if self.use_aspect and self.use_color:
            return image, aspect_ratio, color_mean, label
        elif self.use_aspect:
            return image, aspect_ratio, label
        elif self.use_color:
            return image, color_mean, label
        else:
            return image, label

#  실험 설정 (Base / Aspect / Color / Aspect+Color)
USE_ASPECT = False    # 실험 A → Base / True → 실험 B/D
USE_COLOR = False     # 실험 A → Base / True → 실험 C/D

#  Dataset 정의
train_dataset = CarImageDataset(train_files, class_to_idx, train_transform, use_aspect=USE_ASPECT, use_color=USE_COLOR)
val_dataset = CarImageDataset(val_files, class_to_idx, val_transform, use_aspect=USE_ASPECT, use_color=USE_COLOR)

#  DataLoader 정의
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

import torch.nn as nn
import timm
import torch

class CustomModel(nn.Module):
    def __init__(self, use_aspect, use_color, num_classes):
        super(CustomModel, self).__init__()

        self.use_aspect = use_aspect
        self.use_color = use_color

        #  EfficientNet-B5 backbone
        self.backbone = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0)  # feature extractor
        backbone_out_features = self.backbone.num_features

        #  Meta feature dimension 계산
        meta_features_dim = 0
        if self.use_aspect:
            meta_features_dim += 1  # aspect ratio 1개
        if self.use_color:
            meta_features_dim += 3  # color_mean (R, G, B) 3개

        #  Classifier
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_features + meta_features_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, aspect_ratio=None, color_mean=None):
        # EfficientNet feature
        x = self.backbone(image)

        # Meta features concat
        aux_list = []

        if self.use_aspect:
            aux_list.append(aspect_ratio)

        if self.use_color:
            aux_list.append(color_mean)

        if aux_list:
            aux_features = torch.cat(aux_list, dim=1)
            x = torch.cat([x, aux_features], dim=1)

        # Final classifier
        out = self.classifier(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=396)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# AdamW + weight_decay 추가 추천 (EffNet 계열에 많이 사용)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

from sklearn.model_selection import StratifiedKFold
import copy
import torch
from tqdm import tqdm
import timm
import glob
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전체 jpg 파일
file_list = glob.glob('/content/train/*/*.jpg')

# 클래스명 추출
def extract_class_name_jpg(path):
    return os.path.basename(os.path.dirname(path))

class_names = sorted(set(extract_class_name_jpg(f) for f in file_list))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

print(f" 클래스 수: {len(class_to_idx)}")

#  라벨 생성
labels = [class_to_idx[extract_class_name_jpg(f)] for f in file_list]

#  Aspect Ratio Feature 함수
def compute_aspect_ratio(path):
    with Image.open(path) as img:
        w, h = img.size
        return w / h

# Dominant Color Feature 함수 (간단한 RGB 평균 사용)
def compute_dominant_color(path):
    with Image.open(path).convert("RGB") as img:
        img = img.resize((16, 16))  # 작은 크기로 줄여서 평균 계산
        np_img = np.array(img) / 255.0
        mean_color = np_img.mean(axis=(0, 1))  # R, G, B 평균
        return mean_color  # (3,)

#  Dataset 클래스 정의 (JPG용)
class CarJPGDataset(Dataset):
    def __init__(self, file_list, class_to_idx, transform=None, use_aspect=False, use_color=False):
        self.file_list = file_list
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.use_aspect = use_aspect
        self.use_color = use_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        class_name = extract_class_name_jpg(path)
        label = self.class_to_idx[class_name]

        meta_features = []

        #  Aspect Ratio 추가
        if self.use_aspect:
            ar = compute_aspect_ratio(path)
            meta_features.append(ar)

        # Dominant Color 추가
        if self.use_color:
            color = compute_dominant_color(path)  # (3,)
            meta_features.extend(color.tolist())

        # if meta_features:
        #     meta_features = torch.tensor(meta_features, dtype=torch.float32)
        #     return image, meta_features, label
        # else:
        #     return image, label
        if meta_features:
            meta_features = torch.tensor(meta_features, dtype=torch.float32)
        else:
            meta_features = torch.zeros(3, dtype=torch.float32)  # <= Dummy tensor (0,0,0)

        return image, meta_features, label
# transform 정의
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
#  StratifiedKFold 정의
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#  전략 설정 (전략 C)
EXPERIMENT = "C"  # A / B / C / D

use_aspect = False
use_color = True

print(f"\n🚀 실험 설정: {EXPERIMENT} (Aspect={use_aspect}, Color={use_color})\n")

 5-Fold 루프 시작
for fold, (train_idx, val_idx) in enumerate(skf.split(file_list, labels)):
    print(f"\n==============================")
    print(f"🔁 Fold {fold + 1} / 5")
    print(f"==============================\n")

    #  Fold별 split
    train_files = [file_list[i] for i in train_idx]
    val_files = [file_list[i] for i in val_idx]

    #  Fold별 Dataset & DataLoader
    train_dataset = CarJPGDataset(train_files, class_to_idx, train_transform, use_aspect, use_color)
    val_dataset = CarJPGDataset(val_files, class_to_idx, val_transform, use_aspect, use_color)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    
    #  Fold별 model / criterion / optimizer 초기화
    model = CustomModel(use_aspect=use_aspect, use_color=use_color, num_classes=396)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    #  EarlyStopping 변수 초기화
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    #  Epoch 루프
    for epoch in range(1, 31):
        print(f"\n📌 Fold {fold+1} | Epoch {epoch}")

        # === 학습 ===
        model.train()
        train_loss = 0.0
        train_correct = 0

        loop = tqdm(train_loader, desc=f"Train Fold {fold+1}", leave=False)
        for batch in loop:
            # 전략 C (USE_COLOR=True)
            if use_aspect and use_color:
                X, meta_aspect, meta_color, y = batch
                X, meta_aspect, meta_color, y = X.to(device), meta_aspect.to(device), meta_color.to(device), y.to(device)
                outputs = model(X, aspect_ratio=meta_aspect, color_mean=meta_color)
            elif use_aspect:
                X, meta_aspect, y = batch
                X, meta_aspect, y = X.to(device), meta_aspect.to(device), y.to(device)
                outputs = model(X, aspect_ratio=meta_aspect)
            elif use_color:
                X, meta_color, y = batch
                X, meta_color, y = X.to(device), meta_color.to(device), y.to(device)
                outputs = model(X, color_mean=meta_color)
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                outputs = model(X)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (outputs.argmax(1) == y).sum().item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # === 검증 ===
        model.eval()
        val_loss = 0.0
        val_correct = 0

        val_loop = tqdm(val_loader, desc=f"Valid Fold {fold+1}", leave=False)
        with torch.no_grad():
            for batch in val_loop:
                if use_aspect and use_color:
                    X, meta_aspect, meta_color, y = batch
                    X, meta_aspect, meta_color, y = X.to(device), meta_aspect.to(device), meta_color.to(device), y.to(device)
                    outputs = model(X, aspect_ratio=meta_aspect, color_mean=meta_color)
                elif use_aspect:
                    X, meta_aspect, y = batch
                    X, meta_aspect, y = X.to(device), meta_aspect.to(device), y.to(device)
                    outputs = model(X, aspect_ratio=meta_aspect)
                elif use_color:
                    X, meta_color, y = batch
                    X, meta_color, y = X.to(device), meta_color.to(device), y.to(device)
                    outputs = model(X, color_mean=meta_color)
                else:
                    X, y = batch
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)

                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                val_correct += (outputs.argmax(1) == y).sum().item()
                val_loop.set_postfix(loss=loss.item())

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # === 로그 출력 ===
        print(f" Fold {fold+1} | Epoch {epoch} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f" Fold {fold+1} | Epoch {epoch} | Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # === EarlyStopping ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = f"/content/drive/MyDrive/team_models/EffNetB5_{EXPERIMENT}_fold{fold+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" Best model saved for Fold {fold+1}!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" EarlyStopping patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(" Early stopping triggered.")
                break

    #  Fold 끝나고 Best 모델 로드
    model.load_state_dict(best_model_wts)
    print(f" Fold {fold+1} Best model loaded.\n")

import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from torchvision import transforms
from PIL import Image

#  고정 경로
TEST_DIR = "/content/test"
SAMPLE_SUB_PATH = "/content/sample_submission.csv"
NUM_CLASSES = 396

#  샘플 제출 파일에서 클래스명 추출
sample = pd.read_csv(SAMPLE_SUB_PATH)
column_names = sample.columns.tolist()[1:]  # 'ID' 제외

#  Transform (JPG용)
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#  Dominant Color 계산 함수
def compute_dominant_color(path):
    with Image.open(path).convert("RGB") as img:
        img = img.resize((16, 16))
        np_img = np.array(img) / 255.0
        mean_color = np_img.mean(axis=(0, 1))
        return mean_color  # (3,)

#  테스트용 Dataset (JPG용 + color_mean 포함)
class TestJPGDatasetWithColor(Dataset):
    def __init__(self, img_root, transform=None):
        self.file_list = []
        for file in os.listdir(img_root):
            if file.endswith('.jpg'):
                self.file_list.append(os.path.join(img_root, file))
        self.file_list.sort()

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        image = Image.open(path).convert('RGB')

        # color_mean 계산
        color_mean = compute_dominant_color(path)
        color_mean = torch.tensor(color_mean, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        fname = os.path.basename(path).replace(".jpg", "")
        return image, color_mean, fname

#  DataLoader 고정
test_dataset = TestJPGDatasetWithColor(TEST_DIR, transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=4
)

#  디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  실험 리스트 (전략 C만!)
exp_list = ["C"]

#  비교 결과 저장용 (submission 합치기)
all_submissions = []

#  실험 루프 시작
for exp_name in exp_list:
    print(f"\n==============================")
    print(f" START INFERENCE: EXPERIMENT {exp_name}")
    print(f"==============================\n")

    #  모델 경로 자동 생성
    FOLD_MODEL_PATHS = [
        f"/content/drive/MyDrive/team_models/EffNetB5_{exp_name}_fold1.pth",
        f"/content/drive/MyDrive/team_models/EffNetB5_{exp_name}_fold2.pth",
        f"/content/drive/MyDrive/team_models/EffNetB5_{exp_name}_fold3.pth",
        f"/content/drive/MyDrive/team_models/EffNetB5_{exp_name}_fold4.pth",
        f"/content/drive/MyDrive/team_models/EffNetB5_{exp_name}_fold5.pth",
    ]

    #  앙상블 결과 초기화
    ensemble_outputs = []

    #  Fold 모델 순서대로 추론
    for fold_idx, model_path in enumerate(FOLD_MODEL_PATHS):
        print(f"\nInference with Fold {fold_idx + 1} Model: {model_path}")

        #  반드시 CustomModel 로드 (전략 C)
        model = CustomModel(use_aspect=False, use_color=True, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # fold별 output 저장
        fold_probs = []

        with torch.no_grad():
            for imgs, color_mean, names in tqdm(test_loader, desc=f"🔍 Fold {fold_idx + 1} Inference"):
                imgs = imgs.to(device)
                color_mean = color_mean.to(device)

                outputs = model(imgs, color_mean=color_mean)
                probs = F.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())

        fold_probs = np.concatenate(fold_probs, axis=0)
        ensemble_outputs.append(fold_probs)

    #  앙상블 평균
    ensemble_outputs = np.stack(ensemble_outputs, axis=0)  # (num_folds, num_samples, num_classes)
    mean_outputs = np.mean(ensemble_outputs, axis=0)       # (num_samples, num_classes)

    #  결과 저장
    results = []
    for idx, path in enumerate(test_dataset.file_list):
        fname = os.path.basename(path).replace(".jpg", "")
        row = {"ID": fname}
        row.update({class_name: mean_outputs[idx, i] for i, class_name in enumerate(column_names)})
        results.append(row)

    submission_df = pd.DataFrame(results)
    submission_df = submission_df[["ID"] + column_names]

    #  파일 저장
    SAVE_SUBMISSION_PATH = f"/content/drive/MyDrive/team_models/submission_fold5_ensemble_{exp_name}.csv"
    submission_df.to_csv(SAVE_SUBMISSION_PATH, index=False)

    print(f"\n 앙상블 서브미션 저장 완료: {SAVE_SUBMISSION_PATH}")

    #  비교용으로 all_submissions에 저장
    submission_df["experiment"] = exp_name
    all_submissions.append(submission_df)

#  최종 비교용 DataFrame 만들기
final_compare_df = pd.concat(all_submissions, axis=0)
compare_save_path = "/content/drive/MyDrive/team_models/all_experiments_submission_compare.csv"
final_compare_df.to_csv(compare_save_path, index=False)

print(f"\n🎉 모든 실험 완료! 비교용 CSV 저장됨: {compare_save_path}")

from google.colab import files

#  전략 C 앙상블 결과물 경로
submission_path = "/content/drive/MyDrive/team_models/submission_fold5_ensemble_C.csv"

#  다운로드 실행
files.download(submission_path)

#  전체 비교 CSV 다운로드
compare_path = "/content/drive/MyDrive/team_models/all_experiments_submission_compare.csv"

files.download(compare_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd

#  GPU 메모리 초기화
torch.cuda.empty_cache()

#  1. Teacher 모델 준비
teacher = CustomModel(use_aspect=False, use_color=True, num_classes=396)
teacher.load_state_dict(torch.load("/content/drive/MyDrive/team_models/EffNetB5_C_fold1.pth"))
teacher = teacher.to(device)
teacher.eval()

#  Freeze teacher
for param in teacher.parameters():
    param.requires_grad = False

# 2. Student 모델 준비
student = CustomModel(use_aspect=False, use_color=True, num_classes=396)
student = student.to(device)

#  3. Optimizer, Criterion, Scaler
T_init = 2.0  #  초기 Temperature
T_min = 1.0   #  최소 Temperature
T_decay = 0.95  #  T 줄이는 비율

criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  #  50에포크 기준
scaler = GradScaler()  #  FP16 Mixed Precision

#  4. Dataset & DataLoader
train_dataset = CarJPGDataset(train_files, class_to_idx, train_transform, use_aspect, use_color)
val_dataset = CarJPGDataset(val_files, class_to_idx, val_transform, use_aspect, use_color)  #  Validation 추가

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=True, prefetch_factor=2)

#  5. Stage 2 학습 (Pseudo-Label 기반 Soft Label 학습)
num_epochs = 50
patience = 5
best_val_loss = float('inf')
counter = 0
T = T_init  # 초기 T

for epoch in range(1, num_epochs + 1):
    student.train()
    total_loss = 0.0
    total_samples = 0

    loop = tqdm(train_loader, desc=f"Stage2 Epoch {epoch}")
    for images, meta_features, _ in loop:
        images = images.to(device)
        meta_features = meta_features.to(device)

        with torch.no_grad():
            pseudo_logits = teacher(images, color_mean=meta_features)
            pseudo_soft_labels = F.softmax(pseudo_logits / T, dim=1)

        optimizer.zero_grad()
        with autocast():  #  Mixed Precision
            outputs = student(images, color_mean=meta_features)
            student_log_probs = F.log_softmax(outputs / T, dim=1)
            loss = criterion(student_log_probs, pseudo_soft_labels) * (T * T)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)  #  Gradient Clipping
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / total_samples
    print(f"📚 Stage2 Epoch {epoch}: Train Loss={avg_train_loss:.4f}")

    #  Temperature 스케줄링
    T = max(T_min, T * T_decay)

    #  Validation
    student.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, meta_features, labels in val_loader:
            images = images.to(device)
            meta_features = meta_features.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = student(images, color_mean=meta_features)
                student_log_probs = F.log_softmax(outputs / T, dim=1)
                pseudo_logits = teacher(images, color_mean=meta_features)
                pseudo_soft_labels = F.softmax(pseudo_logits / T, dim=1)
                loss = criterion(student_log_probs, pseudo_soft_labels) * (T * T)

            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    print(f" Validation Loss={avg_val_loss:.4f} | Validation Accuracy={val_acc:.4f}")

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(student.state_dict(), "/content/drive/MyDrive/team_models/Student_stage2_fold1_best.pth")
        print(f" Best model saved at Epoch {epoch}!")
    else:
        counter += 1
        print(f" EarlyStopping patience: {counter}/{patience}")
        if counter >= patience:
            print(" Early stopping triggered.")
            break

    scheduler.step()  #  Learning Rate Scheduler

print(" 학습 완료!")

