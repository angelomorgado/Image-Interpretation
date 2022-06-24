from PIL import Image
import os
import numpy as np

from AuxFiles.postProcessing import morphologicalClose, morphologicalOpen, post_processing

#===============================================================================
# get the path/directory
preds_dir = '../EvaluationDataset/seg_pred_post' # Prediction folder
masks_dir = '../EvaluationDataset/seg_true_masks' # Ground truth folder
#===============================================================================

# Function that returns the iou evaluation of a single image (Receives two numpy arrays)
#This is function exists to be called by other classes
def iouEval(pred_mask, true_mask):
    
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
    
    return np.mean([class0Intersection/class0Count, class1Intersection/class1Count])
    
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

# ==================================== Main ===================================
def main():
    #List of file names
    predList = os.listdir(preds_dir)
    maskList = os.listdir(masks_dir)

    predList.sort()
    maskList.sort()

    # Initialize iou values for each image
    iouList = []
    diceList = []

    for imgName, maskName in zip(predList, maskList):

        pred = Image.open(preds_dir + '/' + imgName).convert('L')
        mask = Image.open(masks_dir + '/' + maskName).convert('L')
        
        #img = np.array(img)
        trueMask = np.array(mask)
        predMask = np.array(pred)

        # Post-processing
        trueMask = post_processing(trueMask)
        predMask = post_processing(predMask)
        predMask = morphologicalClose(predMask)
        predMask = morphologicalOpen(predMask)

        # Evaluate
        imageIoU = iouEval(predMask, trueMask)
        imageDice = dice(predMask, trueMask, 1)

        print('\n====================================================')
        print(f'IoU evaluation for image {imgName}: {imageIoU}')
        print(f'Dice evaluation for image {imgName}: {imageDice}')
        print('====================================================')      

        iouList.append(imageIoU)
        diceList.append(imageDice)


    print('\n\n====================================================')  
    print("Mask R-CNN model:")
    print('====================================================')
    print(f'=        IoU score: {np.mean(iouList)}')
    print(f'=        Dice score: {np.mean(diceList)}')
    print(f'=        The total evaluation is: {np.mean([np.mean(iouList), np.mean(diceList)])}')
    print('====================================================')  

if __name__ == "__main__":
    main()
