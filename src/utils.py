import cv2
import numpy as np

def combine_masks(mask1,mask2):
    combined_mask = cv2.bitwise_or(mask1,mask2)
    return combined_mask

"""
def mask_to_image(frame,mask):
    masked_image = cv2.bitwise_and(frame,frame,mask=mask)

    #set white background
    masked_image = np.where(masked_image[..., None] > 0, frame, 255)
    return masked_image
"""

def mask_to_image(frame, mask):
    #convert mask to 2d
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_expanded = (mask > 0)[..., np.newaxis]
    
    #apply mask
    result = np.where(mask_expanded, frame, 255)
    
    return result.astype(np.uint8)

def crop_image_to_mask(image,mask):
    y_inds,x_inds = np.where(mask>0)

    if len(y_inds) == 0 or len(x_inds) == 0:
        return image

    y_min,y_max = np.min(y_inds), np.max(y_inds)
    x_min,x_max = np.min(x_inds), np.max(x_inds)

    pad = 10
    h, w, _ = image.shape
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    
    # Crop the image to these dimensions
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image



def save_image(image,filepath,name):
    cv2.imwrite(str(filepath)+f"/{name}.png",image)