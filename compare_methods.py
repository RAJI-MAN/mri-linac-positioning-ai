import numpy as np
import torch
from nilearn import datasets
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

# Load MRI
img = datasets.load_mni152_template()
data = img.get_fdata()
slice_index = data.shape[2] // 2
image = data[::4, ::4, slice_index]
image = image / np.max(image)

# ---- Load trained model ----
from ai_shift_model import ShiftNet
model = ShiftNet()
model.eval()

# ---- Function: classical method ----
def classical_shift(original, shifted):
    corr = correlate2d(shifted, original, mode='valid')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    center = np.array(original.shape) // 2
    return np.array([x - center[1], y - center[0]])

# ---- Test loop ----
errors_classical = []
errors_ai = []

for _ in range(20):
    shift_x = np.random.randint(-8, 8)
    shift_y = np.random.randint(-8, 8)

    shifted = np.roll(image, shift_x, axis=0)
    shifted = np.roll(shifted, shift_y, axis=1)

    true_shift = np.array([shift_x, shift_y])

    # Classical
    pred_classical = classical_shift(image, shifted)
    error_classical = np.linalg.norm(pred_classical - true_shift)

    # AI
    tensor = torch.tensor(shifted).unsqueeze(0).unsqueeze(0).float()
    pred_ai = model(tensor).detach().numpy()[0]
    error_ai = np.linalg.norm(pred_ai - true_shift)

    errors_classical.append(error_classical)
    errors_ai.append(error_ai)

# ---- Plot ----
plt.plot(errors_classical, label="Classical")
plt.plot(errors_ai, label="AI")
plt.xlabel("Test Sample")
plt.ylabel("Error")
plt.title("AI vs Classical Shift Detection")
plt.legend()

import os
os.makedirs("results", exist_ok=True)
plt.savefig("results/comparison.png")
plt.show()

print("Average Classical Error:", np.mean(errors_classical))
print("Average AI Error:", np.mean(errors_ai))