import cv2
from torch.utils.data import Dataset

class GTSRBDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["File_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
        image = image[y1:y2, x1:x2]

        if self.transform:
            image = self.transform(image=image)["image"]

        label = row["ClassId"]
        return image, label