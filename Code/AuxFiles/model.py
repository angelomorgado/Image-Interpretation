from torch import true_divide
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(numClasses):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get number of input features for the classifier
    inFeatures = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses)
    
    # Now get the number of input features for the mask classifier
    inFeatures_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hiddenLayer = 256
    
    # And replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(inFeatures_mask, hiddenLayer, numClasses)
    
    # Return the fine-tuned model
    return model