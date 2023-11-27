import os
import numpy as np
import matplotlib.pyplot as plt

depth_directory = 'depth_arrays'  # replace with the path to your directory containing the depth arrays
mask_indices_directory = 'output'  # replace with the path to your directory containing the mask indices arrays

# Loop through the depth arrays in the depth_directory
for filename in os.listdir(depth_directory):
    if filename.endswith('.npy'):
        depth_file_path = os.path.join(depth_directory, filename)
        depth_array = np.load(depth_file_path)
        print(depth_array.shape)
        # Extract the index 'n' from the filename depth_n.npy
        index = filename.split('_')[1].split('.')[0]
        
        # Construct the corresponding mask indices filename and file path
        mask_indices_filename = f'rgb_{index}mask_indices.npy'
        mask_indices_file_path = os.path.join(mask_indices_directory, mask_indices_filename)
        
        if os.path.exists(mask_indices_file_path):
            mask_indices = np.load(mask_indices_file_path)
            
            # Create a blank mask with the same shape as the depth array
            mask = np.zeros_like(depth_array, dtype=np.uint8)
            
            # Set the pixels at the mask indices to 1 (white)
            mask[mask_indices[0], mask_indices[1]] = 1
            
            # Overlay the mask on the depth array
            depth_array = np.where(mask, 1, depth_array)
            
            # Visualize the resulting image
            plt.imshow(depth_array, cmap='gray')
            plt.axis('off')
            
            # Save the figure to the depth_directory
            output_filename = f'depth_{index}_masked.png'
            output_file_path = os.path.join(depth_directory, output_filename)
            plt.savefig(output_file_path)
            plt.close()