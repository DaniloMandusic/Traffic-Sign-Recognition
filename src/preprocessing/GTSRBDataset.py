import cv2
import pandas as pd
from torch.utils.data import Dataset

class GTSRBDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["path"])
        x1, y1, x2, y2 = row["roi"]

        # Crop ROI
        image = image[y1:y2, x1:x2]

        if self.transform:
            image = self.transform(image)

        label = row["label"]
        return image, label