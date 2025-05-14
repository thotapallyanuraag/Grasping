import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the path to your dataset and pick a sample index
dataset_path = "data/cornell-grasp"  # Adjust if necessary
sample_index = 0  # Choose an image index to inspect

# Load the RGB image (adjust based on how the dataset is structured)
rgb_image_path = os.path.join(dataset_path, "rgb", f"rgb_{sample_index}.png")  # Modify path if needed
depth_image_path = os.path.join(dataset_path, "depth", f"depth_{sample_index}.png")  # Modify path if needed

# Load images using PIL or OpenCV
rgb_img = np.array(Image.open(rgb_image_path))
depth_img = np.array(Image.open(depth_image_path))

# Display the RGB image
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title("RGB Image")
plt.axis('off')

# Display the Depth image
plt.subplot(1, 2, 2)
plt.imshow(depth_img, cmap='gray')
plt.title("Depth Image")
plt.axis('off')

plt.show()

# Print out the shapes and sample pixel values for inspection
print(f"RGB Image shape: {rgb_img.shape}")
print(f"Depth Image shape: {depth_img.shape}")
print(f"Sample pixel values in RGB Image: {rgb_img[0, 0, :]}")  # First pixel of RGB image





# import os
# import numpy as np

# dataset_path = "datasets/cornell"  # Adjust if needed

# def is_invalid(npz_path):
#     try:
#         data = np.load(npz_path)
#         return data["quality_map"].size == 0
#     except:
#         return True

# bad_files = []
# for root, _, files in os.walk(dataset_path):
#     for f in files:
#         if f.endswith(".npz"):
#             full_path = os.path.join(root, f)
#             if is_invalid(full_path):
#                 bad_files.append(full_path)

# print(f"Found {len(bad_files)} invalid files.")
# for f in bad_files:
#     os.remove(f)
#     print(f"Removed: {f}")


# # import os
# # import numpy as np
# # from pathlib import Path

# # def clean_cornell_dataset(dataset_path):
# #     # Fix path references
# #     dataset_path = Path(dataset_path)
# #     processed_path = dataset_path / 'processed'
    
# #     # Track problematic files
# #     empty_files = []
# #     missing_files = []
    
# #     # Check all NPZ files
# #     for npz_file in processed_path.glob('*.npz'):
# #         try:
# #             with np.load(npz_file) as data:
# #                 if any(arr.size == 0 for arr in data.values()):
# #                     empty_files.append(npz_file)
# #                     npz_file.unlink()  # Delete empty file
# #         except Exception as e:
# #             print(f"Error processing {npz_file}: {str(e)}")
# #             npz_file.unlink()  # Delete corrupted file
    
# #     # Check for missing files in main directory
# #     for txt_file in dataset_path.glob('*/*cpos.txt'):
# #         prefix = txt_file.name.split('cpos')[0]
# #         npz_file = processed_path / f"{prefix}.npz"
# #         if not npz_file.exists():
# #             missing_files.append((txt_file, npz_file))
    
# #     # Generate report
# #     print(f"\nCleaning complete:")
# #     print(f"- Removed {len(empty_files)} empty/corrupted NPZ files")
# #     print(f"- Found {len(missing_files)} missing NPZ files")
    
# #     if missing_files:
# #         print("\nMissing NPZ files (need regeneration):")
# #         for txt, npz in missing_files[:5]:
# #             print(f"  - Source: {txt}")
# #             print(f"    Should map to: {npz}")

# # # Run the cleaner
# # clean_cornell_dataset('/home/balannag/ggcnn/datasets/cornell')



# # import numpy as np; 
# # d = np.load('datasets/cornell/01/pcd010.npz'); 
# # print('Input shape:', d['rgb'].shape)

# # import numpy as np
# # import os
# # from utils.data.cornell_data import CornellDataset

# # dataset = CornellDataset('datasets/cornell')
# # nan_files = []

# # for i in range(len(dataset.grasp_files)):
# #     try:
# #         rgb = dataset.get_rgb(i)
# #         depth = dataset.get_depth(i)
# #         if np.isnan(rgb).any() or np.isnan(depth).any():
# #             nan_files.append(dataset.rgb_files[i])
# #     except:
# #         continue

# # print(f'Found {len(nan_files)} files with NaN values')
# # if nan_files:
# #     print('First 5 problematic files:')
# #     for f in nan_files[:5]: print(f'  - {f}')