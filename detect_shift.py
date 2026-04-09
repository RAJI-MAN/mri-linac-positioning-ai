from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Load MRI
img = datasets.load_mni152_template()
data = img.get_fdata()

# Get middle slice
slice_index = data.shape[2] // 2
original = data[::2, ::2, slice_index]  

# ---- Create shifted image (same as before) ----
shift_x = 10
shift_y = 5

shifted = np.roll(original, shift_x, axis=0)
shifted = np.roll(shifted, shift_y, axis=1)

# ---- DETECT SHIFT USING CROSS-CORRELATION ----
correlation = correlate2d(shifted, original, mode='same')

# Find peak (best alignment)
y, x = np.unravel_index(np.argmax(correlation), correlation.shape)

# Convert to shift relative to center
center_y, center_x = np.array(correlation.shape) // 2
detected_shift_x = x - center_x
detected_shift_y = y - center_y

# ---- PRINT RESULTS ----
print(f"Actual shift: x={shift_x}, y={shift_y}")
print(f"Detected shift: x={detected_shift_x}, y={detected_shift_y}")

# ---- VISUALISE ----
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(original, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(shifted, cmap='gray')
plt.title("Shifted")

plt.show()
