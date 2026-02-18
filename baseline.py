import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

# ===============================
# SPEED SETTINGS
# ===============================
torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = True  # 🔥 speed boost

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

# ===============================
# CLASS MAP
# ===============================
CLASS_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

# ===============================
# LIGHTER + FASTER MODEL
# ===============================
def get_model():
    return smp.Unet(
        encoder_name="resnet18",  # 🔥 MUCH FASTER
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )

# ===============================
# LOSS
# ===============================
class_weights = torch.tensor(
    [5.0, 5.0, 2.0, 2.0, 10.0, 10.0, 10.0, 5.0, 0.5, 0.1]
).to(DEVICE)

criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
criterion_dice = smp.losses.DiceLoss(mode='multiclass')

def hackathon_combo_loss(pred, mask):
    return 0.5 * criterion_ce(pred, mask) + 0.5 * criterion_dice(pred, mask)

# ===============================
# DATASET
# ===============================
class DesertDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [
            f for f in os.listdir(images_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        new_mask = np.zeros_like(mask, dtype=np.int64)
        for actual_id, train_id in CLASS_MAP.items():
            new_mask[mask == actual_id] = train_id

        return image, torch.from_numpy(new_mask).long()

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])

    IMAGES_PATH = r"C:\Users\ASUS\Desktop\Datasets\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    MASKS_PATH = r"C:\Users\ASUS\Desktop\Datasets\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"

    full_dataset = DesertDataset(IMAGES_PATH, MASKS_PATH, train_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,          # 🔥 bigger batch
        shuffle=True,
        num_workers=2,         # 🔥 parallel loading
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = get_model().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    scaler = torch.cuda.amp.GradScaler()  # 🔥 AMP for speed

    EPOCHS = 15  # 🔥 reduce epochs for speed
    best_iou = 0.0

    print(f"🚀 Fast Training on {DEVICE}")

    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # 🔥 mixed precision
                preds = model(images)
                loss = hackathon_combo_loss(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_iou_total = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                preds = model(images)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    preds.argmax(dim=1).unsqueeze(1),
                    masks.unsqueeze(1),
                    mode='multiclass',
                    num_classes=NUM_CLASSES
                )

                batch_iou = smp.metrics.iou_score(
                    tp, fp, fn, tn, reduction="macro"
                )

                val_iou_total += batch_iou.item()

        avg_val_iou = val_iou_total / len(val_loader)

        print(f"Epoch {epoch+1} → Train Loss: {avg_train_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), "best_model_weights.pth")
            print("🌟 Saved Best Model")

    print(f"🏆 Done | Best Val IoU: {best_iou:.4f}")
