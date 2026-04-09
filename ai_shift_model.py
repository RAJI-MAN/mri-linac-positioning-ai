import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nilearn import datasets

# ----------------------------
# LOAD MRI
# ----------------------------
img = datasets.load_mni152_template()
data = img.get_fdata()

slice_index = data.shape[2] // 2
image = data[::4, ::4, slice_index]   # downsample (important)

# Normalize
image = image / np.max(image)

# ----------------------------
# CREATE TRAINING DATA
# ----------------------------
def create_sample(img):
    shift_x = np.random.randint(-8, 8)
    shift_y = np.random.randint(-8, 8)

    shifted = np.roll(img, shift_x, axis=0)
    shifted = np.roll(shifted, shift_y, axis=1)

    return shifted, np.array([shift_x, shift_y], dtype=np.float32)

X, y = [], []

for _ in range(300):
    img_shifted, shift = create_sample(image)
    X.append(img_shifted)
    y.append(shift)

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X = torch.tensor(X).unsqueeze(1).float()
y = torch.tensor(y).float()

# ----------------------------
# MODEL (FIXED VERSION)
# ----------------------------
class ShiftNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),   # 🔥 KEY FIX
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = ShiftNet()

# ----------------------------
# TRAINING
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(10):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ----------------------------
# TEST MODEL
# ----------------------------
test_img, true_shift = create_sample(image)

test_tensor = torch.tensor(test_img).unsqueeze(0).unsqueeze(0).float()

pred_shift = model(test_tensor).detach().numpy()[0]

print("\nTrue shift:", true_shift)
print("Predicted shift:", pred_shift)