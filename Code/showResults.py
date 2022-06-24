from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

#===========================================================================================
# Images
input_img_path = '../EvaluationDataset/seg_input'
pred_path = '../EvaluationDataset/seg_pred_noPost' 
post_pred_path = '../EvaluationDataset/seg_pred_post' 
alphapose_path = '../EvaluationDataset/alphapose_pred' 
results_path = '../EvaluationDataset/final_results'

#===========================================================================================

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

#===========================================================================================

# Get the the list of file names
imgList = os.listdir(input_img_path)

for imgName in tqdm(imgList, desc='imgName'):
    
    fig = plt.figure(figsize=(10,7))

    rows = 2
    columns = 2
    
    # Original Image
    original_img = Image.open(input_img_path + '/' + imgName).convert('RGB')
    fig.add_subplot(rows, columns, 1)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title('Original Image')
    
    # Alphapose
    alphapose_img = Image.open(alphapose_path + '/' + imgName).convert('RGB')
    fig.add_subplot(rows, columns, 2)
    plt.imshow(alphapose_img)
    plt.axis('off')
    plt.title('Predicted alphapose image')
    
    # Prediction without post-processing
    pred_img = Image.open(pred_path + '/' + imgName).convert('L')
    fig.add_subplot(rows, columns, 3)
    plt.imshow(pred_img)
    plt.axis('off')
    plt.title('Predicted mask without any post-processing')
    
    # Prediction after post-processing
    pred_img_post = Image.open(post_pred_path + '/' + imgName).convert('L')
    fig.add_subplot(rows, columns, 4)
    plt.imshow(pred_img_post)
    plt.axis('off')
    plt.title('Predicted mask after post-processing')
    
    # Save final result
    fig_img = fig2img(fig).convert('RGB')
    fig_img.save(results_path + '/' + imgName)
    fig_img.close()
    
print("Finished organizing data. You can check the results in " + '/' + results_path.split('/')[-2] + '/' + results_path.split('/')[-1])
    
    