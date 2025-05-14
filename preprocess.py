# import os
# import numpy as np
# from PIL import Image

# os.makedirs('datasets/cornell/01', exist_ok=True)
# for subdir in [f'datasets/cornell-grasp/{i:02d}' for i in range(1, 11)]:
#     for f in os.listdir(subdir):
#         if f.endswith('r.png'):
#             rgb = np.array(Image.open(f'{subdir}/{f}'))
#             rgb = Image.fromarray(rgb).resize((300, 300))
#             rgb = np.array(rgb)
#             np.savez(f'data/processed/{f[:-6]}.npz', rgb=rgb)
# print(f'Processed {len(os.listdir("data/processed"))} files')



#different one
# import numpy as np; 
# d = np.load('datasets/cornell/01/pcd010.npz'); 
# print('File keys:', d.files); 
# print('RGB shape:', d['rgb'].shape)


#different one
# import numpy as np; 
# d=np.load('datasets/cornell/01/pcd010.npz'); 
# print('New RGB shape:', d['rgb'].shape)



import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def safe_process_image(rgb_path, output_dir):
    try:
        # Load and validate image
        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Invalid image dimensions")
        
        # Convert to CHW format
        rgb = np.transpose(rgb, (2, 0, 1))
        
        # Save processed file
        out_path = os.path.join(output_dir, os.path.basename(rgb_path).replace('r.png', '.npz'))
        np.savez(out_path, rgb=rgb)
        
        # Create minimal annotation file
        with open(out_path.replace('.npz', 'cpos.txt'), 'w') as f:
            f.write("0 0 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 0")  # Default rectangle
    
    except Exception as e:
        print(f"Skipping {rgb_path}: {str(e)}")
        return False
    return True

# Process all images
input_dir = 'data/cornell-grasp'  # Your original dataset
output_dir = 'datasets/cornell/01'
os.makedirs(output_dir, exist_ok=True)

processed = 0
for subdir in [f'{input_dir}/{i:02d}' for i in range(1, 11)]:
    for f in tqdm(os.listdir(subdir)):
        if f.endswith('r.png'):
            if safe_process_image(os.path.join(subdir, f), output_dir):
                processed += 1

print(f"Successfully processed {processed} images")