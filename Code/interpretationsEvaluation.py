from tqdm import tqdm
import os

pred_path = "../EvaluationDataset/prediction_results/"
truth_path = "../EvaluationDataset/prediction_ground_truth/"
num_carac = 12

def __handlePredFile(pred_file):
    pred = []
    
    # Read the first two lines of the file that are not relevant
    s = pred_file.readline()
    s = pred_file.readline()
    
    # Get the characteristics of the image in the list
    for i in range(num_carac):
        s = pred_file.readline().split(': ')[1]
        s = s.split('\n')[0]
        pred.append(s)
    return pred

def __handleTruthFile(truth_file):
    truth = []
    
    # Get the characteristics of the image in the list
    for i in range(num_carac):
        s = truth_file.readline()
        s = s.split('\n')[0]
        truth.append(s)
    return truth

def __getRatio(pred, truth):
    count = 0
    for i in range(num_carac):
        if pred[i] == truth[i]:
            count += 1
    
    return count / num_carac

def main():
    name_list = os.listdir(pred_path)
    
    evaluation_list = []
    
    for name in name_list:
        print(name)
        print('\n======================================================================\n')
        pred_file = open(pred_path + name, 'r')
        truth_file = open(truth_path + name, 'r')
        
        pred = __handlePredFile(pred_file)
        truth = __handleTruthFile(truth_file)
        
        ratio = __getRatio(pred, truth)
        
        print(f'The ratio of the correct predictions in the image {name.split(".txt")[0]} is {ratio}')
        evaluation_list.append(ratio)
    
    print('\n\n======================================================================')
    print(f'= The average ratio of the correct predictions is {sum(evaluation_list) / len(evaluation_list)}')
    print('======================================================================\n')
        

        
if __name__ == '__main__':
    main()
    
    
    