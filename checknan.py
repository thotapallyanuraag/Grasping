# import os
# import numpy as np

# def contains_nan(file_path):
#     try:
#         with open(file_path, 'r') as f:
#             values = [float(x) for x in f.read().split()]
#         arr = np.array(values)
#         return np.isnan(arr).any()
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return True  # treat unreadable files as bad

# # Path to your cleaned dataset
# dataset_root = 'data/cornell-grasp'
# bad_files = []

# for subdir, _, files in os.walk(dataset_root):
#     for f in files:
#         if f.endswith('cpos.txt'):
#             full_path = os.path.join(subdir, f)
#             if contains_nan(full_path):
#                 bad_files.append(full_path)

# print(f"\nTotal files with NaN values: {len(bad_files)}")
# for bf in bad_files:
#     print(bf)

# import os
# import glob

# invalid_files = []
# valid_files = []

# root = 'data/cornell-grasp'

# for txt_path in glob.glob(os.path.join(root, '**', '*cpos.txt'), recursive=True):
#     try:
#         with open(txt_path, 'r') as f:
#             content = f.read()
#         # Try converting to float
#         values = [float(x) for x in content.split()]
#         if len(values) % 8 != 0:
#             print(f"[FORMAT ERROR] Not divisible by 4: {txt_path} (count={len(values)})")
#             invalid_files.append(txt_path)
#             continue
#         # Passed all checks
#         valid_files.append(txt_path)
#     except ValueError as ve:
#         print(f"[VALUE ERROR] Invalid float in: {txt_path} — {ve}")
#         invalid_files.append(txt_path)
#     except Exception as e:
#         print(f"[READ ERROR] Could not read: {txt_path} — {e}")
#         invalid_files.append(txt_path)

# print("\n✅ Summary:")
# print(f"Total valid files   : {len(valid_files)}")
# print(f"Total invalid files : {len(invalid_files)}")


# import cv2, glob
# for f in glob.glob('data/cornell-grasp/**/*r.png', recursive=True):
#     img = cv2.imread(f)
#     if img is None or img.size == 0 or img.shape[0] < 100 or img.shape[1] < 100 or img.shape[2] != 3:
#         print(f'[INVALID] {f} — shape: {None if img is None else img.shape}')


#checking rgb files integrity
# import cv2
# import numpy as np
# import glob

# TARGET_SHAPE = (3, 300, 300)

# print("🔍 Scanning all r.png files for pixel validity and shape consistency...\n")

# for f in glob.glob('data/cornell-grasp/**/*r.png', recursive=True):
#     try:
#         img = cv2.imread(f, cv2.IMREAD_COLOR)
#         if img is None or img.size == 0:
#             print(f"[UNREADABLE or EMPTY] {f}")
#             continue

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         if np.isnan(img).any():
#             print(f"[NaN PIXELS] {f}")
#             continue

#         if np.all(img == 0):
#             print(f"[ALL ZEROS] {f}")
#             continue

#         # Resize to (640, 480) as expected by GG-CNN
#         img = cv2.resize(img, (640, 480))
#         img = np.transpose(img, (2, 0, 1))  # HWC → CHW

#         # Simulate crop (centered 300x300)
#         c, h, w = img.shape
#         top = (h - 300) // 2
#         left = (w - 300) // 2
#         img = img[:, top:top+300, left:left+300]

#         # Final shape check
#         if img.shape != TARGET_SHAPE:
#             print(f"[SHAPE ERROR] {f} — got {img.shape}")

#     except Exception as e:
#         print(f"[PROCESSING ERROR] {f} — {e}")


# #checking depth integrity
# import os
# from PIL import Image
# import numpy as np

# root = 'data/cornell-grasp'
# invalid_files = []
# total_files = 0

# print("🔍 Checking all .d.tiff files for readability and pixel validity...\n")

# for dirpath, _, filenames in os.walk(root):
#     for fname in filenames:
#         if fname.endswith('d.tiff'):
#             total_files += 1
#             path = os.path.join(dirpath, fname)
#             try:
#                 img = np.array(Image.open(path))
#                 if img.size == 0:
#                     print(f"[EMPTY] {path}")
#                     invalid_files.append(path)
#                     continue
#                 if np.isnan(img).any():
#                     print(f"[NaN PIXELS] {path}")
#                     invalid_files.append(path)
#                     continue
#                 if np.all(img == 0):
#                     print(f"[ALL ZEROS] {path}")
#                     invalid_files.append(path)
#             except Exception as e:
#                 print(f"[UNREADABLE] {path} — {e}")
#                 invalid_files.append(path)

# print("\n✅ Summary:")
# print(f"Total .d.tiff files checked: {total_files}")
# print(f"Invalid or problematic files : {len(invalid_files)}")


import tensorboardX
import torch

print(torch.__version__)
print(tensorboardX.__version__)