from AuxFiles.model import get_model_instance_segmentation
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import AuxFiles.transforms as T
import torch
import numpy as np
import os
import io
import json

import AuxFiles.postProcessing as pp

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


#========================================================
# Images
inputImg_path = '/mnt/c/Users/angel/codigos/Projeto/EvaluationDataset/seg_input'
pred_path = '/mnt/c/Users/angel/codigos/Projeto/EvaluationDataset/seg_pred_noPost' # Just to store the pred masks to evaluation
post_pred_path = '/mnt/c/Users/angel/codigos/Projeto/EvaluationDataset/seg_pred_post' # Just to store the pred masks to evaluation

# Model
modelPath = 'AuxFiles/RCNNmodel.pth'

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#========================================================

print('Loading model...')

# Load model
model = get_model_instance_segmentation(2)
model.load_state_dict(torch.load(modelPath))
model = model.to(device)
model.eval()


#List of file names
imgList = os.listdir(inputImg_path)

segmentation_dict = {}

area_list = []

for imgName in tqdm(imgList):
    print(imgName)

    fig = plt.figure(figsize=(10,7))

    rows = 2
    columns = 2

    # OriginalImg
    original_img = Image.open(inputImg_path + '/' + imgName).convert('RGB')

    imgT = transforms.ToTensor()(original_img).unsqueeze_(0).to(device)

    h, w = imgT.shape[2:]

    with torch.no_grad():
        prediction = model(imgT)
    

    maskList = prediction[0]['masks'][:, 0].mul(255).byte().cpu().numpy()

    pred_mask = np.zeros((h, w))
    
    for i in maskList:
        pred_mask += i

    #Save the image without any pre-processing
    no_pre_process_image = Image.fromarray(pred_mask).convert('L')
    no_pre_process_image.save(pred_path + '/' + imgName)

    # Post-processing
    # Adjust the masks
    pred_mask[pred_mask > 180] = 255
    pred_mask[pred_mask <= 180] = 0

    pred_mask = pp.morphologicalClose(pred_mask, 10)
    pred_mask = pp.morphologicalOpen(pred_mask, 5)

    pred_image = Image.fromarray(pred_mask).convert('L') # Full mask in Image format

    pred_image.save(post_pred_path + '/' + imgName)
    
    maskList = pp.removeDuplicates(maskList) # Removes any duplicates
    
    area_list.append(np.count_nonzero(pred_mask) / (h * w))
    segmentation_dict[imgName] = maskList.tolist()

alpha = np.mean(area_list)

with open('AuxFiles/areaValue.txt', 'w') as alpha_file:
    alpha_file.write(str(alpha))
    
print('Prediction completed!')
print('Storing into a json, may take a while...')
with open('../EvaluationDataset/seg_pred_noPost/segmentation_result.json', 'w') as fp:
    json.dump(segmentation_dict, fp)
    
print('Results stored in EvaluationDataset/seg_pred_noPost/segmentation_result.json')
