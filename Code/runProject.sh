# PreProcessing
#python3 preProcessing.py

# Run alphapose
python3 alphapose.py --indir ../EvaluationDataset/seg_input --outdir ../EvaluationDataset/alphapose_pred --vis_fast --save_img --sp

# Run Mask-RCNN
python3 predict.py

# Run Data organization
python3 showResults.py

# Run Data Interpretation
python3 interpret.py