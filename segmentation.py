import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os 
def show_anns(anns, ax=None):
    # Show a list of segmentation masks in the image
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if not ax:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    print(sorted_anns)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.4]])
        img[m] = color_mask
    
    ax.imshow(img)

def show_ann(ann, box, image, color_mask=None):
    # Show a single segmentation mask in the image
    if len(ann) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    color = np.concatenate([[255,0,0], [0.6]])
    m = ~ann['segmentation']
    
    m[:2, :], m[-2:, :], m[:, :2], m[:, -2:] = False, False, False, False
    m = np.pad(m, pad_width=((box[0],image.shape[0]-box[0]-m.shape[0]),(box[2],image.shape[1]-box[2]-m.shape[1])),mode='constant',constant_values=False)
    if color_mask is not None:
        print("The pixel number close to the white is: ", np.count_nonzero(color_mask))
        m = np.logical_and(m, color_mask)
    indices = np.where(m==True)
    
    img[m] = color
    ax.imshow(img)
    return indices

def crop_image(image, box):
    # box = [ymin, ymax, xmin, xmax]
    return image[box[0]:box[1], box[2]:box[3]]
def color_filter(image, tagert_color, threshold=40):
    # tagert_color = [r,g,b]
    image_float = image.astype(np.float32)
    distances = np.linalg.norm(image_float - tagert_color, axis=-1)  # [h, w]
    mask = distances < threshold

    return mask

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    sam_checkpoint_path = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
    )
    def process_and_save(image_path, output_dir):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        # image = cv2.resize(image, (width, height))
        crop_shape = [0., 1.0, 0.4, 1.0]
        box =   [int(crop_shape[0]*height), int(crop_shape[1]*height), int(crop_shape[2]*width), int(crop_shape[3]*width)]
        cropped_image = crop_image(image, box)
        masks = mask_generator.generate(cropped_image)

        print("There are a total number of {} masks".format(len(masks)))
        print(image.shape)
        plt.figure()
        plt.imshow(image)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

        color_mask = color_filter(image, [255,255,255])
        indices = show_ann(sorted_anns[0],box, image, color_mask)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, os.path.basename(image_path)))
        np.save(os.path.join(output_dir, os.path.basename(image_path).split(".")[0]+'mask_indices.npy'), indices)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dir_path = "rgb_images"
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)
        process_and_save(image_path, output_dir)
        print("Finish processing {}".format(image_name))

    torch.cuda.empty_cache()
