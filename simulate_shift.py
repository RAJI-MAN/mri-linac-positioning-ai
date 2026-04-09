from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# Load MRI
img = datasets.load_mni152_template()
data = img.get_fdata()

# Select middle slice
slice_index = data.shape[2] // 2
original = data[:, :, slice_index]

# ---- SIMULATE SHIFT ----
shift_x = 10   # pixels (left/right)
shift_y = 5    # pixels (up/down)

shifted = np.roll(original, shift_x, axis=0)
shifted = np.roll(shifted, shift_y, axis=1)

# ---- PLOT ----
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(original, cmap="gray")
plt.title("Original (Correct Position)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(shifted, cmap="gray")
plt.title("Shifted (Wrong Position)")
plt.axis("off")

plt.show()

print(f"Simulated shift: x={shift_x}, y={shift_y}")