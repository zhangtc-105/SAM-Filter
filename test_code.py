import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    print(sorted_anns)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    image_path = "color.png"
    image = cv2.imread(image_path)
    

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_point = np.array([[230, 366],[363,369]])
    input_label = np.array([1,1])
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label,multimask_output=True)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    
        indices = np.where(mask==True)
        numpy_repo = "perception_wire_pred/indices"+str(i)+".npy"
        np.save(numpy_repo, indices)
        plt.axis('off')
        pic_repo = "segementation_result_"+str(i)+".png"
        plt.savefig(pic_repo)
    # mask_input = predictor.get_mask_input(input_point)

    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(image)

    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()

