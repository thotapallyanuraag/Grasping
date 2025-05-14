import os
import argparse
import logging
import numpy as np
from PIL import Image
import torch.utils.data
import matplotlib.pyplot as plt
from models import get_network
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset

logging.basicConfig(level=logging.INFO)

def prepare_image(img):
    """Convert image to matplotlib-compatible format"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # Handle channel ordering
    if img.ndim == 3:
        if img.shape[0] in [1, 3]:  # Channels-first (C,H,W)
            img = np.transpose(img, (1, 2, 0))  # Convert to (H,W,C)
        elif img.shape[2] in [1, 3]:  # Already channels-last
            pass
        else:
            raise ValueError(f"Unrecognized image shape: {img.shape}")
    
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    return img




def visualize_results(rgb_img, depth_img, q_img, ang_img, width_img, output_dir, filename):
    """Visualize RGB, depth and predicted grasps"""
    fig = plt.figure(figsize=(15, 5))
    
    # RGB Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb_img)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # Depth Image
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(depth_img, cmap='gray')
    ax2.set_title('Depth Image')
    ax2.axis('off')
    
    # Grasp Predictions
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(q_img, cmap='jet', alpha=0.5)
    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=args.n_grasps)
    for g in grasps:
        g.plot(ax3, color='red')
    ax3.set_title('Predicted Grasps')
    ax3.axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close(fig)





def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                       help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    # Corrected argument checks
    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args

def visualize_results(rgb_img, depth_img, q_img, ang_img, width_img, output_dir, filename):
    """Visualize RGB, depth and predicted grasps."""
    fig = plt.figure(figsize=(15, 5))
    
    # RGB Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb_img)
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # Depth Image
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(depth_img, cmap='gray')
    ax2.set_title('Depth Image')
    ax2.axis('off')
    
    # Grasp Predictions
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(q_img)
    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    for g in grasps:
        g.plot(ax3)
    ax3.set_title('Predicted Grasps')
    ax3.axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

if __name__ == '__main__':
    args = parse_args()

    # Load Network
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    ggcnn = get_network('ggcnn')
    net = ggcnn(input_channels=input_channels)
    net.load_state_dict(torch.load(args.network))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=args.augment, random_zoom=args.augment,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                          lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                  no_grasps=args.n_grasps,
                                                  grasp_width=width_img)
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.jacquard_output:
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                with open(jo_fn, 'a') as f:
                    for g in grasps:
                        f.write(test_data.dataset.get_jname(didx) + '\n')
                        f.write(g.to_jacquard(scale=1024 / 300) + '\n')

            
            if args.vis:
                try:
                    # Get images
                    rgb_img = test_data.dataset.get_rgb(didx, rot, zoom)
                    depth_img = test_data.dataset.get_depth(didx, rot, zoom)
                    
                    # Convert images
                    rgb_img = prepare_image(rgb_img)
                    depth_img = prepare_image(depth_img)
                    
                    # Create figure
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Plot RGB
                    ax1.imshow(rgb_img)
                    ax1.set_title('RGB Input')
                    ax1.axis('off')
                    
                    # Plot Depth (handle single channel)
                    if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                        depth_img = depth_img.squeeze(-1)
                    ax2.imshow(depth_img, cmap='gray')
                    ax2.set_title('Depth Input')
                    ax2.axis('off')
                    
                    # Plot Grasps
                    ax3.imshow(q_img, cmap='jet', alpha=0.7)
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=args.n_grasps)
                    for g in grasps:
                        g.plot(ax3, color='red')
                    ax3.set_title('Grasp Predictions')
                    ax3.axis('off')
                    
                    # Save figure
                    os.makedirs("output/visuals", exist_ok=True)
                    output_path = os.path.join("output/visuals", f"grasp_{didx.item()}.png")
                    plt.savefig(output_path, bbox_inches='tight', dpi=100)
                    plt.close(fig)
                    
                except Exception as e:
                    logging.error(f"Visualization failed for sample {didx.item()}")
                    logging.error(f"Error: {str(e)}")
                    logging.error(f"RGB shape: {rgb_img.shape if 'rgb_img' in locals() else 'N/A'}")
                    logging.error(f"Depth shape: {depth_img.shape if 'depth_img' in locals() else 'N/A'}")
                    if 'fig' in locals():
                        plt.close(fig)









    if args.iou_eval:
        total = results['correct'] + results['failed']
        if total > 0:
            accuracy = results['correct'] / total
            logging.info('IOU Results: %d/%d = %f' % (results['correct'], total, accuracy))
        else:
            logging.info('No grasps detected for evaluation.')

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))

