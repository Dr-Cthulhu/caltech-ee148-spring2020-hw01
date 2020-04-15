import os
import numpy as np
import json
from PIL import Image

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 

with open(os.path.join(preds_path,'preds.json'),'r') as f:
    preds = json.load(f)
    # get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

for i in range(len(file_names)):
    print("predictions for image " + str(i+1))

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)

    I_bounded = np.copy(I)

    # draw boxes
    boxes = preds[file_names[i]]
    for box in boxes:
        for row in range(box[0], box[2]):
            I_bounded[row][box[1]] = [0, 255, 0]
            I_bounded[row][box[3]] = [0, 255, 0]
        for col in range(box[1], box[3]):
            I_bounded[box[0]][col] = [0, 255, 0]
            I_bounded[box[2]][col] = [0, 255, 0]

    # display image
    bounded_image = Image.fromarray(I_bounded, 'RGB')
    # bounded_image.show()
    bounded_image.save(os.path.join(preds_path, 'RL_bounded_' + str(i) + '.jpg'))

