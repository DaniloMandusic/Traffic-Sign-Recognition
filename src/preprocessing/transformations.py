from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_transforms(image_size, train=True):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=15),
            A.RandomBrightnessContrast(),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])