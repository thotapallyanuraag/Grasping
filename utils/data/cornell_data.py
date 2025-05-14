import os
import glob
import numpy as np
import cv2

from utils.dataset_processing.image import Image
import numpy as np
from .grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        #print(f"[DEBUG] Found {len(graspf)} grasp file in {file_path}")
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        self.grasp_files = []
        self.depth_files = []
        self.rgb_files = []

        count = 0

        for g, d, r in zip(graspf, depthf, rgbf):
            if not (os.path.exists(g) and os.path.exists(d) and os.path.exists(r)):
                continue
            if os.path.getsize(g) == 0 or os.path.getsize(d) == 0 or os.path.getsize(r) == 0:
                continue
            try:
                grs = grasp.GraspRectangles.load_from_cornell_file(g)
                if len(grs.grs) == 0:
                    continue
                # Extra: check if image loads and resizes properly
                img = cv2.imread(r)
                if img is None or img.size == 0 or len(img.shape)!= 3 or img.shape[2]!=3:
                    continue
                # img = cv2.resize(img, (640, 480))
                # img = np.transpose(img, (2, 0, 1))
                # c, h, w = img.shape
                # top = (h - 300) // 2
                # left = (w - 300) // 2
                # cropped = img[:, top:top+300, left:left+300]
                # if cropped.shape != (3, 300, 300):
                #     continue
            except Exception as e:
                print(f"[SKIPPED] {g} - {e}")
                continue

            self.grasp_files.append(g)
            self.depth_files.append(d)
            self.rgb_files.append(r)

            count +=1
            #print(f"[DEBUG] Total valid samples loaded: {count}")





        l = len(self.grasp_files)
        if l == 0:
            raise FileNotFoundError('No valid Cornell samples found after filtering.')

        # Apply start/end cropping
        self.grasp_files = self.grasp_files[int(l*start):int(l*end)]
        self.depth_files = self.depth_files[int(l*start):int(l*end)]
        self.rgb_files = self.rgb_files[int(l*start):int(l*end)]


    def _get_crop_attrs(self, idx):
        try:
            gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
            if len(gtbbs.grs) == 0:
                raise ValueError("No valid grasps in annotation")
            center = gtbbs.center
            left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
            top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
            return center, left, top
        except Exception as e:
            # Return center of image if grasps are invalid
            return (240, 320), 160, 120

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        try:
            gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
            center, left, top = self._get_crop_attrs(idx)
            gtbbs.rotate(rot, center)
            gtbbs.offset((-top, -left))
            gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
            return gtbbs
        except:
            return grasp.GraspRectangles([])

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0):
        try:
            import cv2
            img_path = self.rgb_files[idx]
            img = cv2.imread(img_path)

            if img is None:
                raise ValueError("Image is None or unreadable.")
            if img.size == 0:
                raise ValueError("Image is zero-size.")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to (640, 480) before transpose
            img = cv2.resize(img, (640, 480))

            # Now transpose HWC → CHW
            img = np.transpose(img, (2, 0, 1))

            # Validate after transpose: [3, 480, 640]
            if img.shape[0] != 3 or img.shape[1] < 300 or img.shape[2] < 300:
                raise ValueError(f"Invalid RGB shape after transpose: {img.shape}")

            rgb_img = image.Image(img.copy())

            center, left, top = self._get_crop_attrs(idx)
            rgb_img.rotate(rot, center)
            rgb_img.crop((top, left),
                        (min(480, top + self.output_size),
                        min(640, left + self.output_size)))
            rgb_img.zoom(zoom)
            rgb_img.resize((self.output_size, self.output_size))

            result = rgb_img.img
            # Final safety check
            if result.shape != (3, self.output_size, self.output_size):
                raise ValueError(f"Final image shape incorrect: {result.shape}")
            return result

        except Exception as e:
            #print(f"Warning: Error processing {self.rgb_files[idx]}: {str(e)}")
            return np.zeros((3, self.output_size, self.output_size), dtype=np.float32)



