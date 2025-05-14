
import os

import numpy as np

from PIL import Image



# CONFIGURE THESE PATHS

input_base = 'data/cornell-grasp'  # CHANGE THIS to your dataset location

output_dir = 'datasets/cornell/01' 



os.makedirs(output_dir, exist_ok=True)

processed = 0



for subdir in [f'{input_base}/{i:02d}' for i in range(1, 11)]:

    if not os.path.exists(subdir):

        print(f"⚠️ Missing: {subdir}")

        continue

        

    for f in os.listdir(subdir):

        if f.endswith('r.png'):

            try:

                rgb = np.array(Image.open(f'{subdir}/{f}').resize((300, 300)))

                np.savez(f'{output_dir}/{f[:-6]}.npz', rgb=rgb)

                processed += 1

            except Exception as e:

                print(f"❌ Failed {f}: {str(e)}")



print(f"✅ Processed {processed} files to {output_dir}")

