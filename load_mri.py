from nilearn import datasets
import matplotlib.pyplot as plt

# Load MRI directly (no nib.load)
img = datasets.load_mni152_template()

data = img.get_fdata()

print("MRI shape:", data.shape)

# Show middle slice
slice_index = data.shape[2] // 2

plt.imshow(data[:, :, slice_index], cmap="gray")
plt.title("MRI Slice")
plt.axis("off")
plt.show()