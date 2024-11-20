#imports
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import cv2
import nnunetv2
from nnunetv2.unnunet.uncertainty_utils import *
import os
import shutil
from PIL import Image

#$ this script is used to compute the uncertainty score for a given dataset, after prediction was done.

#proba_dir ='' ### the folder with the checkpoints folders (output of previous script)
#raw_path ='' ##path to the folder with the dataset the user wants to predict ( input of previous script)
#labels = ''## optional - path to the labels of the dataset'
#score_type = '' ## optional - the score type to use for the uncertainty score. default is 'class_entropy' - other options are 'total_entropy' and 't_test'.
#output_pred_path = '' ## optional - path to the folder where the predictions will be saved. default is 'proba_dir + '/unnunet_pred''
def run_uncertainty_on_fold(proba_dir, raw_path, score_type , labels , output_pred_path = False, use_predictions = 'all'):
    dice_list = []
    uncertainty_scores = []
    name_list = [name.split('.')[0].replace('_0000','') for name in os.listdir(raw_path) if '_0000' in name]

    for image_name in name_list:
        #compute p values map for the image
        #$ map all dir in the folder - those are the checkpoints.
        if use_predictions == 'all':
            checkpoint_list = [checkpoint for checkpoint in os.listdir(proba_dir)
                            if os.path.isdir(os.path.join(proba_dir, checkpoint)) and 'checkpoint' in checkpoint] # using all checkpoints
        elif use_predictions == 'best':
            checkpoint_list = [checkpoint for checkpoint in os.listdir(proba_dir)
                            if os.path.isdir(os.path.join(proba_dir, checkpoint)) and 'checkpoint_best' in checkpoint] # using only the 5 best checkpoints (one from each fold)
        else:
            raise ValueError(
                f"Invalid value for 'use_predictions': '{use_predictions}'. "
                "Expected values are 'all' or 'best'."
            )
            
        background_array = [] #$ background
        tumor_array = [] #$ tumor
        dhgp_array = [] #$ dhgp
        liver_array = [] #$ liver
        #$ for each checkpoint we will have a list of probability maps for each class.

        for checkpoint in checkpoint_list:
            #$ Load probs from .npz file
            print('loading npz file')
            prediction_file = np.load(proba_dir + '/' + checkpoint + '/' + image_name + '.npz', allow_pickle=True)
            print('npz file loaded')
            #$ append the probability map of each class to the class array.
            background_array.append(prediction_file['probabilities'][0, :, :])
            tumor_array.append(prediction_file['probabilities'][1, :, :])
            dhgp_array.append(prediction_file['probabilities'][2, :, :])
            liver_array.append(prediction_file['probabilities'][3, :, :])

            print(f'npz shape = {background_array[0].shape}')

        #$ convert the class arrays to numpy arrays.
        background_array = np.array(background_array)
        tumor_array = np.array(tumor_array)
        dhgp_array = np.array(dhgp_array)
        liver_array = np.array(liver_array)

        print(background_array.shape)
        print(tumor_array.shape)
        print(dhgp_array.shape)
        print(liver_array.shape)

        foreground_mean = np.mean(np.mean(np.stack([tumor_array, dhgp_array, liver_array], axis=0), axis=0), axis = 0)
        mask = (foreground_mean.T > 0.3).astype(np.uint8) #$ changed this to incorporate all three foreground classes

        map = np.zeros_like(mask)
        if score_type == 't_test':
            p_values_map = T_test_on_single_image(background_array, tumor_array, dhgp_array, liver_array, plot_results = False) #$ leftoff
            uncertainty_score = uncertainty_from_mask_and_valmap(p_values_map ,  mask)
            map = p_values_map

        elif score_type == 'class_entropy':
            class_entropy_map = entropy_map_fun(np.mean(background_array,axis = 0), np.mean(tumor_array,axis = 0), np.mean(dhgp_array,axis = 0), np.mean(liver_array,axis = 0))
            uncertainty_score =  uncertainty_from_mask_and_valmap(class_entropy_map ,  mask)
            map = class_entropy_map

        elif score_type == 'total_entropy':
            #append class one and class two on axis 0
            np.concatenate((background_array, tumor_array, dhgp_array, liver_array), axis = 0)
            total_entropy_map = -np.sum(background_array * np.log(background_array), axis=0)
            uncertainty_score =  uncertainty_from_mask_and_valmap(total_entropy_map ,  mask)
            map = total_entropy_map

        uncertainty_scores.append(uncertainty_score)
        if labels:
            label =  load_niigii_file(labels + '/' + image_name + '.nii.gz')
            temp_dice =   dice(mask , label)
            dice_list.append(temp_dice)
        
        if not output_pred_path:
            output_pred_path = proba_dir + '/unnunet_pred'
        if not os.path.exists(output_pred_path):
            os.makedirs(output_pred_path)

        # Load the predicted and artifact masks
        predicted_mask = Image.open(proba_dir + '/best_prediction/' + image_name + '.png')
        artifact_mask = Image.open(raw_path + '/' + image_name + '_mask.png')

        # Convert images to NumPy arrays
        predicted_array = np.array(predicted_mask)
        artifact_array = np.array(artifact_mask)

        # Ensure both arrays are of the same shape (resize or pad if necessary)
        if predicted_array.shape != artifact_array.shape:
            raise ValueError("Predicted and artifact masks must have the same dimensions.")

        # Element-wise multiplication of the two masks
        result_array = predicted_array * artifact_array

        # Convert the result back to an image
        result_image = Image.fromarray(result_array)

        # Save the result to the output directory
        result_image.save(output_pred_path + '/' + image_name + '_postprocessed.png')
        
        #save the uncertainty map
        normalized_map = (map - np.min(map)) / (np.max(map) - np.min(map)) * 255
        normalized_map = normalized_map.astype(np.uint8)
    
        # Load and downscale the artifact mask by a factor of 16
        artifact_mask = Image.open(raw_path + '/' + image_name + '_mask.png')
        downscaled_artifact_mask = artifact_mask.resize(
            (artifact_mask.width // 16, artifact_mask.height // 16),
            Image.NEAREST  # Use nearest neighbor to preserve binary values
        )

        # Convert the downscaled artifact mask and uncertainty map to NumPy arrays
        artifact_array = np.array(downscaled_artifact_mask)
        uncertainty_array = np.array(normalized_map)  # Assuming normalized_map is already a NumPy array

        # Ensure both arrays have the same shape
        if artifact_array.shape != uncertainty_array.shape:
            raise ValueError("Uncertainty map and downscaled artifact mask must have the same dimensions.")

        # Element-wise multiplication of the downscaled artifact mask and uncertainty map
        result_array = artifact_array * uncertainty_array

        # Save the resulting map
        uncertaintypng_path = f"{output_pred_path}/{image_name}_uncertainty_map.png"
        print(f'saving uncertainty map to {uncertaintypng_path}')
        plt.imsave(uncertaintypng_path, result_array, cmap='viridis')

    #save the uncertainty scores, with the image names , if dice availible also dice scores.
    uncertainty_df = pd.DataFrame({'image_name': name_list, 'uncertainty_score': uncertainty_scores})
    if labels:
        uncertainty_df['dice_score'] = dice_list
    uncertainty_path = output_pred_path + '/uncertainty_scores.csv'
    print(f'saving uncertainty values to {uncertainty_path}')
    uncertainty_df.to_csv(uncertainty_path, index=False)
    return uncertainty_df

        

def run_uncertainty_on_fold_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proba_dir', type=str, default='', help='path to the folder with the checkpoints folders (output of previous script)')
    parser.add_argument('--raw_path', type=str, default='', help='path to the folder with the dataset the user wants to predict ( input of previous script)')
    parser.add_argument('--labels', type=str, default='', help='optional - path to the labels of the dataset')
    parser.add_argument('--score_type', type=str, default='class_entropy', help='optional - the score type to use for the uncertainty score. default is class_entropy - other options are total_entropy and t_test')
    parser.add_argument('--output_pred_path', type=str, default='', help='optional - path to the folder where the predictions will be saved. default is proba_dir + /unnunet_pred')
    parser.add_argument('--use_predictions', type=str, default='all', help="optional - options are 'all' and 'best', referring to if you want to use all calculated predictions for the uncertainty or only the 5 best models (which are used for the prediction).")
    args = parser.parse_args()

    
    run_uncertainty_on_fold(args.proba_dir, args.raw_path, args.score_type , args.labels , args.output_pred_path, args.use_predictions)

if __name__ == '__main__':
    run_uncertainty_on_fold_entry()


