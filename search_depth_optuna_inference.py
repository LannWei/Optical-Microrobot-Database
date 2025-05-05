import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from thop import profile

robot_name = "robot_18_TwinRobot2"
# "robot_8_robot" "robot_10_eight_ball" "robot_12_BallHolderStableBot" 
# "robot_14_new_FourBallRobot" "robot_16_SphericalArm" "robot_18_TwinRobot2"
config_path = f"./checkpoint_NAS_depth/robot_8_robot/optuna_best_config.csv"
os.makedirs(f"./checkpoint_NAS_depth/{robot_name}", exist_ok=True)
best_model_path = f"./checkpoint_NAS_depth/{robot_name}/best_model_retrained.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best config
cfg = pd.read_csv(config_path).iloc[0].to_dict()

# Dataset
class DepthDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(row['depth'], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# Model
class SearchableCNN(nn.Module):
    def __init__(self, num_conv, num_filters, fc_dim, dropout_rate, use_bn):
        super().__init__()
        layers = []
        in_channels = 3
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = num_filters
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_dim, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# Load data
data_root = f"./Image_Depth/{robot_name}"
txt_file = [f for f in os.listdir(data_root) if f.endswith(".txt")][0]
depths = []
with open(os.path.join(data_root, txt_file), 'r') as f:
    for line in f:
        parts = line.strip().split()
        depth = float(parts[-1])
        depths.append(depth)

image_files = sorted([f for f in os.listdir(data_root) if f.lower().endswith(('jpg', 'png'))])
image_paths = [os.path.join(data_root, f) for f in image_files]
df = pd.DataFrame({"image_path": image_paths, "depth": depths})

train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_ds = DepthDataset(train_df)
val_ds = DepthDataset(val_df)
test_ds = DepthDataset(test_df)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, drop_last=True)

# Train
def train_model(model, loader, val_loader, optimizer, criterion, epochs=40):
    best_mse = float("inf")
    for epoch in range(epochs):
        model.train()
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

        # Eval on val
        model.eval()
        preds, targets_val = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                pred = model(imgs).squeeze().cpu().numpy()
                preds.extend(pred)
                targets_val.extend(targets.numpy())
        mse = mean_squared_error(targets_val, preds)
        print(f"Epoch {epoch+1}, Val MSE: {mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), best_model_path)

# Build & Train model
model = SearchableCNN(
    int(cfg["num_conv"]), int(cfg["num_filters"]), int(cfg["fc_dim"]),
    float(cfg["dropout_rate"]), bool(cfg["use_bn"])
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"])) #float(cfg["lr"])
criterion = nn.MSELoss()

train_model(model, train_loader, val_loader, optimizer, criterion)

# Evaluate best model on test set
model.load_state_dict(torch.load(best_model_path))
model.eval()
preds, targets = [], []
with torch.no_grad():
    for imgs, t in test_loader:
        imgs = imgs.to(device)
        pred = model(imgs).squeeze().cpu().numpy()
        preds.extend(pred)
        targets.extend(t.numpy())

mse = mean_squared_error(targets, preds)
r2 = r2_score(targets, preds)
macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),), verbose=False)

results = []
results.append({
        "MSE": mse,
        "R2": r2,
        "Params_M": round(params / 1e6, 2),
        "FLOPs_G": round(macs / 1e9, 2)})

print(f"\n✅ Final Test - MSE: {mse:.4f}, R2: {r2:.4f}, Params: {params / 1e6:.2f}M, FLOPs: {macs / 1e9:.2f}G")

output_csv = f"./checkpoint_NAS_depth/{robot_name}/results_{robot_name}.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n📊 Results saved to {output_csv}")