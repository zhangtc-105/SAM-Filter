import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

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

def show_ann(ann, box, image):
    # Show a single segmentation mask in the image
    if len(ann) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0

    m = ~ann['segmentation']
    m[:2, :], m[-2:, :], m[:, :2], m[:, -2:] = False, False, False, False
    m = np.pad(m, pad_width=((box[0],image.shape[0]-box[0]-m.shape[0]),(box[2],image.shape[1]-box[2]-m.shape[1])),mode='constant',constant_values=False)
    indices = np.where(m==True)
    np.save("mask_indices.npy", indices)
    color_mask = np.concatenate([[255,0,0], [0.5]])
    img[m] = color_mask
    ax.imshow(img)

def crop_image(image, box):
    # box = [ymin, ymax, xmin, xmax]
    return image[box[0]:box[1], box[2]:box[3]]

if __name__ == "__main__":

    image = cv2.imread("Images/color.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image = cv2.resize(image, (width//2, height//2))
    height, width, _ = image.shape
    crop_shape = [0., 1.0, 0.1, 0.9]
    box = [int(crop_shape[0]*height), int(crop_shape[1]*height), int(crop_shape[2]*width), int(crop_shape[3]*width)]
    croped_image = crop_image(image,box)

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
    masks = mask_generator.generate(croped_image)
    print("There are a total number of {} masks".format(len(masks)))
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    show_ann(sorted_anns[0],box, image)
    plt.axis('off')
    plt.savefig("mask_result.png")
    torch.cuda.empty_cache()
