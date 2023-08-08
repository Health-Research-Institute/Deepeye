# Import libraries
import numpy as np
import torch
from torchvision import transforms


#Create Image Stack
def create_image_stack(images, labels, image_paths, nImLevelsData):
    
    image_paths = [img_path.split("\\")[1] for img_path in image_paths]

    image_count = len(images) // nImLevelsData  

    stacked_data = []
    stacked_labels = []
    stacked_image_names = []

    normalize = transforms.Normalize([0.485], [0.229]) 

    for i in range(image_count):
        start_index = i * nImLevelsData
        end_index = start_index + nImLevelsData

        single_stack = images[start_index+1:end_index] # only 8 images without background image

        image_stack = np.stack([normalize(torch.from_numpy(img.astype(np.float32))) for img in single_stack])
        image_stack = torch.from_numpy(image_stack)
        
        stacked_data.append(image_stack)
        stacked_labels.append(labels[start_index])
        stacked_image_names.append(image_paths[start_index+1:end_index])
    
    return stacked_data, stacked_labels, stacked_image_names


#Define Tiles
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

#Class Encoding
def create_classEncoding(classes):
    Encoding = {}
    for i,class_ in enumerate(classes):
        
        tensor_values = [0] * len(classes)
        tensor_values[i] = 1
        Encoding[class_] = torch.FloatTensor(tensor_values)

    return Encoding



