Grasping CNN

The is a lightweight, fully-convolutional network which predicts the quality and pose of antipodal grasps at every pixel in an input depth image.  The lightweight and single-pass generative nature allows for fast execution and closed-loop control, enabling accurate grasping in dynamic environments where objects are moved during the grasp attempt.

Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets

Currently, the [Cornell Grasping Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp) is supported.

### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

Trained models are saved in `output/models` by default, with the validation score appended.
