from tqdm import tqdm
from PIL import Image
import os
import json
import numpy as np
import AuxFiles.postProcessing as pp
import AuxFiles.interpretations as i
import AuxFiles.textInterpretations as ti

#===============================================================================

#Alphapose Json
alphapose_results_path = "../EvaluationDataset/alphapose_pred/alphapose-results.json"

#Segmentation Json
segmentation_results_path = "../EvaluationDataset/seg_pred_noPost/segmentation_result.json"

#Results dir
results_dir = "../EvaluationDataset/interpretation_results/"

#===============================================================================

def main():
    # Load Jsons
    #Alphapose
    print("Loading alphapose json...")
    with open(alphapose_results_path) as json_file:
        alphapose_results = json.load(json_file)
    
    # Segmentation
    print("Loading segmentation json...")
    print("This process can take a while (~30 seconds for 20 images)")
    with open(segmentation_results_path) as json_file:
        segmentation_results = json.load(json_file)
    
    print("All jsons were successfully loaded!\n")
    
    # Adapt the data
    print("adjusting data...")
    for k in segmentation_results.keys():
        segmentation_results[k] = pp.post_processing(np.array(segmentation_results[k]))
        #segmentation_results[k] = pp.morphologicalOpen(segmentation_results[k])
        #segmentation_results[k] = pp.morphologicalClose(segmentation_results[k])
    
    # ================================ Predictions =======================================
    print('================================ Predictions =======================================\n')
    # Go through all images in the folder
    for k in segmentation_results.keys():
        f = open(f'../EvaluationDataset/prediction_results/{k}.txt', 'w')
        prediction_text = ''
        
        print(f'Predictions for image {k}:\n')
        f.write(f'Predictions for image {k}:\n\n')
        
        # Get the number of people
        number_of_people = i.getPeopleQuantity(list(segmentation_results[k]))
        print(f'Number of people: {number_of_people}')
        f.write(f'Number of people: {number_of_people}\n')
        
        # Concatenate the information to the prediction text
        prediction_text += ti.numberOfPeople(number_of_people, k)
          
        # Get the number of people front/back/side to the camera
        facing_camera, facing_back, facing_sides, masked_faces = i.getPeopleOrientation_maskedFaces(k, alphapose_results)
        print(f'People facing the camera: {facing_camera}')
        f.write(f'People facing the camera: {facing_camera}\n')
        print(f'People with the back to the camera: {facing_back}')
        f.write(f'People with the back to the camera: {facing_back}\n')
        print(f'People sideways to the camera: {facing_sides}')
        f.write(f'People sideways to the camera: {facing_sides}\n')
        
        # Concatenate the information to the prediction text
        prediction_text += ti.cameraOrientation(number_of_people, (facing_camera, facing_back, facing_sides))
        
        # Get the number of people standing/sitting/laying
        standing_people, sitting_people, laying_people = i.getPeoplePose(k, alphapose_results)
        print(f'People standing: {standing_people}')
        f.write(f'People standing: {standing_people}\n')
        print(f'People sitting: {sitting_people}')
        f.write(f'People sitting: {sitting_people}\n')
        print(f'People laying: {laying_people}')
        f.write(f'People laying: {laying_people}\n')
        
        # Concatenate the information to the prediction text
        prediction_text += ti.peoplePose(number_of_people, (standing_people, sitting_people, laying_people))
        
        # Get the number of people close/far to the camera
        far_people, close_people = i.getPeopleDistance(list(segmentation_results[k]))
        print(f'People far from the camera: {far_people}')
        f.write(f'People far from the camera: {far_people}\n')
        print(f'People close to the camera: {close_people}')
        f.write(f'People close to the camera: {close_people}\n')
        
        # Concatenate the information to the prediction text
        prediction_text += ti.peopleDistance(number_of_people, (close_people, far_people))
        
        # Get the number of people with the face covered by something (e.g., a mask)
        print(f'People with their faces covered (mask): {masked_faces}')
        f.write(f'People with their faces covered (mask): {masked_faces}\n')
        
        # Concatenate the information to the prediction text
        prediction_text += ti.peopleMasked(number_of_people, masked_faces)
        
        # Get shirt tones
        dark_tone, light_tone = i.getShirtTones(k, alphapose_results) 
        print(f'Most common shirt tones: {"dark shirts" if dark_tone > light_tone else "light shirts"}') 
        f.write(f'Most common shirt tones: {"dark" if dark_tone > light_tone else "light"}\n')
        
        if dark_tone > light_tone:
            tone = 'dark'
        elif dark_tone < light_tone:
            tone = 'light'
        else:
            tone = 'both'
        
        prediction_text += ti.shirtTone(number_of_people, tone)
        
        # Get jean tones
        dark_tone, light_tone = i.getJeanTones(k, alphapose_results)
        print(f'Most common pants tones: {"dark pants" if dark_tone > light_tone else "light pants"}') 
        f.write(f'Most common pants tones: {"dark" if dark_tone > light_tone else "light"}\n\n')
        
        if dark_tone > light_tone:
            tone = 'dark'
        elif dark_tone < light_tone:
            tone = 'light'
        else:
            tone = 'both'
        
        prediction_text += ti.pantsTone(number_of_people, tone)
        
        f.write('Image generated description:\n\n')
        f.write(prediction_text)
    
        print('=============================================================')
        
    
         

if __name__ == "__main__":
    main()
    
    
    