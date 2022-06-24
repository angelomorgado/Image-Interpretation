import cv2
import numpy as np

def __iouEval(pred_mask, true_mask):
    
    h, w = pred_mask.shape
    
    class1Count = 0
    class1Intersection = 0
    class0Count = 0
    class0Intersection = 0

    # IOU Evaluation
    for y in range(h):
        for x in range(w):
            # Class 0
            if true_mask[y,x] == 0:
                class0Count += 1

                if true_mask[y,x] == pred_mask[y,x]:
                    class0Intersection += 1
            
            # Class 1
            if true_mask[y,x] == 1:
                class1Count += 1

                if true_mask[y,x] == pred_mask[y,x]:
                    class1Intersection += 1
                    
    x = np.mean([class0Intersection/class0Count, class1Intersection/class1Count])
    return x


# Erosion followed by dilation
def morphologicalOpen(maskArray, value=10):
    kernel = np.ones((value,value),np.uint8)
    img = cv2.morphologyEx(maskArray, cv2.MORPH_OPEN, kernel)
    return img

# Dilation followed by erosion
def morphologicalClose(maskArray, value=10):
    kernel = np.ones((value,value),np.uint8)
    img = cv2.morphologyEx(maskArray, cv2.MORPH_CLOSE, kernel)
    return img

# Normalizes the image and increases the accuracy
def post_processing(img_array):
    img_array[img_array > 200] = 255
    img_array[img_array <= 200] = 0
    
    # If the image isn't already normalized
    if np.amax(img_array > 1):
        img_array = img_array/255
    
    return img_array

# Eliminates duplicates using iouEval
def removeDuplicates(segmentation_results):   
    for k in segmentation_results:
        for i, k1 in enumerate(segmentation_results):
            if (k != k1).any() and __iouEval(k, k1) > 0.54:
                segmentation_results = np.delete(segmentation_results,i, axis=0)
                print('Eliminated 1 duplicate')
                break
    return np.array(segmentation_results)