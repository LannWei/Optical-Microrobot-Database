import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Regex for PX_RY
pose_pattern = re.compile(r"P(\d+)_R(\d+)")
image_extensions = (".png", ".jpg", ".jpeg")

class PoseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.P_map = {v: i for i, v in enumerate(sorted(self.df['P'].unique()))}
        self.R_map = {v: i for i, v in enumerate(sorted(self.df['R'].unique()))}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)
        p_label = self.P_map[row['P']]
        r_label = self.R_map[row['R']]
        return image, p_label, r_label

    def __len__(self):
        return len(self.df)

# Load and split from image folder
def load_dataframes(root_path="./Image_Pose/robot_1_four_ball"):
    data = []
    label_map = {}
    label_idx = 0

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if not os.path.isdir(folder_path):
            continue

        match = pose_pattern.match(folder)
        if not match:
            continue

        p_val, r_val = int(match.group(1)), int(match.group(2))
        pose_label = f"P{p_val}_R{r_val}"

        if pose_label not in label_map:
            label_map[pose_label] = label_idx
            label_idx += 1

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, fname)
                data.append({
                    "image_path": image_path,
                    "P": p_val,
                    "R": r_val,
                    "pose_class": label_map[pose_label]
                })

    df = pd.DataFrame(data)
    # train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['pose_class'], random_state=42)
    # val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['pose_class'], random_state=42)
    num_p = df['P'].nunique()
    num_r = df['R'].nunique()
    # return train_df, val_df, test_df, num_p, num_r
    return df, num_p, num_r