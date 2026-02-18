import os
import torch
import cv2
import numpy as np
from baseline import get_model, CLASS_MAP
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- SETTINGS ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_weights.pth"

TEST_IMAGE_DIR = r"C:\Users\ASUS\Desktop\Datasets\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images"
TEST_MASK_DIR = r"C:\Users\ASUS\Desktop\Datasets\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation"  # if GT available
SAVE_DIR = "test_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = len(CLASS_MAP)

# ---------------- DATASET ----------------
class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image = transformed["image"]

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, img_name)
            mask = cv2.imread(mask_path, 0)
            return image, torch.tensor(mask, dtype=torch.long), img_name

        return image, img_name

# ---------------- IOU FUNCTION ----------------
def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious)

# ---------------- MAIN ----------------
def run_test():
    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = TestDataset(TEST_IMAGE_DIR, TEST_MASK_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_iou = []

    with torch.no_grad():
        for image, mask, name in loader:
            image = image.to(DEVICE)
            mask = mask.numpy()

            output = model(image)
            pred = torch.argmax(output, dim=1).cpu().numpy()

            iou = compute_iou(pred[0], mask[0], NUM_CLASSES)
            total_iou.append(iou)

            # Save prediction
            save_path = os.path.join(SAVE_DIR, name[0])
            cv2.imwrite(save_path, pred[0].astype(np.uint8))

    print("Mean IoU:", np.mean(total_iou))

if __name__ == "__main__":
    run_test()
