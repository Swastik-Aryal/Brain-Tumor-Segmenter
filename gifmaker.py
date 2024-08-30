import imageio
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from skimage.transform import rotate
import nibabel as nib
from tqdm import tqdm
import argparse
import sys

# custom colors for the mask overlay
colors = [
    [0, 0, 0, 0],  # Background (fully transparent)
    [0, 0, 1, 1],  # Necrotic/Core (blue)
    [1, 1, 0, 1],  # Edema (yellow)
    [1, 0, 0, 1]   # Enhancing (red)
]

# Create a discrete colormap with these colors
custom_cmap = ListedColormap(colors)


def create_gif(image, mask_image = None, gif_path = None,fps = 18, padding_axial = 40):
    '''
    padding_axial : Number of black frames to add at the beginning of the axial slice (adjust as needed)
    
    '''
    # get name
    fs = image.split('.')[:-1][0]
    
    # Check if the images are in nii or npy format
    
    if image.split('.')[-1] in ["nii", "nii.gz"]:
        image = nib.load(image).get_fdata()
    else:
        image = np.load(image)
    if mask_image is not None:
        if mask_image.split('.')[-1] in ["nii", "nii.gz"]:
            mask_image = nib.load(mask_image).get_fdata()
        else:
            mask_image = np.load(mask_image)
    
    frames = []
    max_slices = max(image.shape)


    # Create the frames
    for i in tqdm(range(max_slices + padding_axial)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')  # Set background to black
        
        # Axial view
        if i >= padding_axial:
            slice_idx = i - padding_axial
            if slice_idx < image.shape[2]:
                axial_slice = image[:, :, slice_idx]
                
                if mask_image is not None:
                    mask_axial_slice = mask_image[:, :, slice_idx]
                    masked_axial = np.ma.masked_where(mask_axial_slice == 0, mask_axial_slice)
                    
                axes[0].imshow(axial_slice, cmap='gray', interpolation='none')
                
                if mask_image is not None:
                    axes[0].imshow(masked_axial, cmap=custom_cmap, alpha=0.6, interpolation='none')
            else:
                axes[0].imshow(np.zeros_like(image[:, :, 0]), cmap='gray', interpolation='none')
        else:
            axes[0].imshow(np.zeros_like(image[:, :, 0]), cmap='gray', interpolation='none')

        # Coronal view
        if i < image.shape[1]:
            coronal_slice = image[:, i, :]
            if mask_image is not None:
                mask_coronal_slice = mask_image[:, i, :]
                masked_coronal = np.ma.masked_where(mask_coronal_slice == 0, mask_coronal_slice)
            axes[1].imshow(rotate(coronal_slice, -90, resize=True), cmap='gray', interpolation='none')
            
            if mask_image is not None:
                axes[1].imshow(rotate(masked_coronal, -90, resize=True), cmap=custom_cmap, alpha=0.6, interpolation='none')
        else:
            axes[1].imshow(np.zeros_like(image[:, :, 0]), cmap='gray', interpolation='none')

        # Sagittal view
        if i < image.shape[0]:
            sagittal_slice = image[i, :, :]
            
            if mask_image is not None:
                mask_sagittal_slice = mask_image[i, :, :]
                masked_sagittal = np.ma.masked_where(mask_sagittal_slice == 0, mask_sagittal_slice)
                
            axes[2].imshow(rotate(sagittal_slice, 90, resize=True), cmap='gray', interpolation='none')
            
            if mask_image is not None:
                axes[2].imshow(rotate(masked_sagittal, 90, resize=True), cmap=custom_cmap, alpha=0.6, interpolation='none')
        else:
            axes[2].imshow(np.zeros_like(image[:, :, 0]), cmap='gray', interpolation='none')

        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('black')  # Set the axes background to black
        
        # Save the frame to a buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    print("Saving.... ")
    # Save the frames as a GIF
    if gif_path is None:
        gif_path = f'{fs}.gif'
        if mask_image is not None:
            gif_path = f'{fs}_masked.gif'
        
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)

    print(f'GIF saved as {gif_path}')
    
def print_instructions():
    print("Usage: python gifmaker.py --image [image path] --mask [mask path] --fps [frames per second]")
    print("Note:")
    print("     --image (Required) path to the unsegmented volume (preferred Flair)")
    print("     --mask (Optional)(default: None) path to segmentation mask.")
    print("     --save_path (Optional)(default: cwd/[image_name]_[mask_status].gif)")
    print("     --fps (Optional)(default: 18) fps for the gif")
    print("     --padding (Optional)(default: 40) black frames to add at the beginning of the axial slice ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create gifs")
    parser.add_argument('--image', type=str, required=True, help="Path to the image file (required).")
    parser.add_argument('--mask', type=str, default=None, help="Path to the mask file (optional).")
    parser.add_argument('--save_path', type=str, default=None, help="Path to the save the gif (optional, default is cwd/[image_name]_[mask_status].gif).")
    parser.add_argument('--fps', type=int, default=18, help="Frames per second (optional, default is 18).")
    parser.add_argument("--padding", type=int, default=40, help="black frames to add at the beginning of the axial slice(optional, default is 40)")

    args = parser.parse_args()

    # Check if the required --image argument is provided
    if not args.image:
        print("Error: The --image argument is required.")
        print_instructions()
        sys.exit(1)
    
    create_gif(args.image, args.mask, args.save_path, args.fps, args.padding)
    
        
    
