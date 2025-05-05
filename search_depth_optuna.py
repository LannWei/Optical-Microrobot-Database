import os
import torch
import torch.nn as nn
import optuna
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from thop import profile

robot_name = "robot_10_eight_ball"
# "robot_8_robot" "robot_10_eight_ball" "robot_12_BallHolderStableBot" 
# "robot_14_new_FourBallRobot" "robot_16_SphericalArm" "robot_18_TwinRobot2"
data_root = f"./Image_Depth/{robot_name}"
os.makedirs(f"./checkpoint_NAS_depth/{robot_name}", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images and depth
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
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_ds = DepthDataset(train_df)
val_ds = DepthDataset(val_df)
test_ds = DepthDataset(test_df)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Objective
best_model_path = f"./checkpoint_NAS_depth/{robot_name}/best_model.pth"
def objective(trial):
    num_conv = trial.suggest_int("num_conv", -3, 7)
    num_filters = trial.suggest_int("num_filters", 64, 256)
    fc_dim = trial.suggest_int("fc_dim", 128, 1024)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    use_bn = trial.suggest_categorical("use_bn", [True])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = SearchableCNN(num_conv, num_filters, fc_dim, dropout_rate, use_bn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, drop_last=True)

    for epoch in range(3):
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, t in val_loader:
            imgs = imgs.to(device)
            pred = model(imgs).squeeze().cpu().numpy()
            preds.extend(pred)
            targets.extend(t.numpy())

    mse = mean_squared_error(targets, preds)
    trial.set_user_attr("val_mse", mse)
    return mse

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Final evaluation
best_trial = study.best_trial
print("\n🏆 Best Configuration:")
for k, v in best_trial.params.items():
    print(f"{k}: {v}")
print(f"Best Val MSE: {best_trial.user_attrs['val_mse']:.4f}")

# Retrain on best config
bp = best_trial.params
model = SearchableCNN(bp["num_conv"], bp["num_filters"], bp["fc_dim"], bp["dropout_rate"], bp["use_bn"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=bp["lr"])
criterion = nn.MSELoss()
train_loader = DataLoader(train_ds, batch_size=bp["batch_size"], shuffle=True, drop_last=True)

for epoch in range(20):
    model.train()
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs).squeeze()
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), best_model_path)

# Test
model.eval()
preds, targets = [], []
with torch.no_grad():
    for imgs, t in DataLoader(test_ds, batch_size=32):
        imgs = imgs.to(device)
        pred = model(imgs).squeeze().cpu().numpy()
        preds.extend(pred)
        targets.extend(t.numpy())

mse = mean_squared_error(targets, preds)
r2 = r2_score(targets, preds)
macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),), verbose=False)

print(f"\n🧪 Test MSE: {mse:.4f}, R2: {r2:.4f}")
print(f"📦 Params: {params / 1e6:.2f}M, FLOPs: {macs / 1e9:.2f}G")

pd.DataFrame([{**bp, "test_mse": mse, "test_r2": r2, "params_m": round(params/1e6, 2), "flops_g": round(macs/1e9, 2)}]).to_csv(
    f"./checkpoint_NAS_depth/{robot_name}/optuna_best_config.csv", index=False)
