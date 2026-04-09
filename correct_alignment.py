from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Load MRI
img = datasets.load_mni152_template()
data = img.get_fdata()

# Get slice
slice_index = data.shape[2] // 2
original = data[::2, ::2, slice_index]

# ---- Simulate shift ----
shift_x = 10
shift_y = 5

shifted = np.roll(original, shift_x, axis=0)
shifted = np.roll(shifted, shift_y, axis=1)

# ---- Detect shift ----
correlation = correlate2d(shifted, original, mode='valid')
y, x = np.unravel_index(np.argmax(correlation), correlation.shape)

center_y, center_x = np.array(original.shape) // 2
detected_shift_x = x - center_x
detected_shift_y = y - center_y

# ---- Correct alignment ----
corrected = np.roll(shifted, -detected_shift_x, axis=0)
corrected = np.roll(corrected, -detected_shift_y, axis=1)

# ---- Plot ----
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(shifted, cmap='gray')
plt.title("Shifted")

plt.subplot(1,3,3)
plt.imshow(corrected, cmap='gray')
plt.title("Corrected")

plt.show()

# ---- Print ----
print(f"Actual shift: x={shift_x}, y={shift_y}")
print(f"Detected shift: x={detected_shift_x}, y={detected_shift_y}")