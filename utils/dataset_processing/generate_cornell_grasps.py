import os
import numpy as np
import cv2
from skimage.draw import polygon
from tqdm import tqdm

def load_grasps(file_path):
    try:
        with open(file_path, 'r') as f:
            values = [float(x) for x in f.read().split()]
        arr = np.array(values).reshape(-1, 4, 2)
        if np.isnan(arr).any():
            return None
        return arr
    except:
        return None

def generate_grasp_map(grasps, img_size=300):
    quality_map = np.zeros((img_size, img_size))
    angle_map = np.zeros((img_size, img_size))
    width_map = np.zeros((img_size, img_size))
    for grasp in grasps:
        center = np.mean(grasp, axis=0).astype(int)
        dx, dy = grasp[1, 0] - grasp[0, 0], grasp[1, 1] - grasp[0, 1]
        angle = np.arctan2(dy, dx)
        width = np.sqrt(dx**2 + dy**2)
        rr, cc = polygon(grasp[:, 1], grasp[:, 0])
        valid = (rr >= 0) & (rr < img_size) & (cc >= 0) & (cc < img_size)
        quality_map[rr[valid], cc[valid]] = 1.0
        angle_map[rr[valid], cc[valid]] = angle
        width_map[rr[valid], cc[valid]] = width
    return quality_map, angle_map, width_map

def preprocess_dataset(input_dir, output_dir, img_size=300):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for f in tqdm(files):
            if f.endswith('r.png'):
                base = f.replace('r.png', '')
                rgb_path = os.path.join(root, f)
                txt_path = os.path.join(root, base + 'cpos.txt')

                if not os.path.exists(txt_path):
                    continue

                grasps = load_grasps(txt_path)
                if grasps is None:
                    continue

                rgb_img = cv2.imread(rgb_path)
                if rgb_img is None:
                    continue
                rgb_img = cv2.resize(rgb_img, (img_size, img_size))

                quality, angle, width = generate_grasp_map(grasps, img_size)
                np.savez(os.path.join(output_dir, base + '.npz'),
                         rgb=rgb_img,
                         quality_map=quality,
                         angle_map=angle,
                         width_map=width)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cornell_path', type=str, required=True, help='Path to Cornell dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    preprocess_dataset(args.cornell_path, args.output_path)


