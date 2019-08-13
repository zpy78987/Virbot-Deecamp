from util.data_api import list_all_fm_file
import numpy as np
import pickle
from dotmap import DotMap

def load_data(filepath):
    inputs, labels, names = data_prepare(filepath)

    data = DotMap()
    data.inputs = inputs
    data.labels = labels
    data.names = names
    
    return data
    
def data_prepare(filepath):
    wav_xs = list_all_fm_file(filepath, 'wav_x')
    inputs = []
    labels = []
    names = []

    print(len(wav_xs))
 
    for i in range(np.min([len(wav_xs),1000])):     #for i in range(len(fm_ys)):     # 437
        wav_x = wav_xs[i]
        fm_y = wav_xs[i].replace('wav_x', 'fm_y')
        name = fm_y.replace('.fm_y', '').split("/", 2)[2]

        with open(wav_x,'rb') as f:
            result_x = pickle.load(f)
        with open(fm_y,'rb') as f:
            result_y = pickle.load(f)
        inputs.append(result_x[:3600, :])
        labels.append(result_y[:3600, :])
        names.append(name)
    return inputs, labels, names  # input: [437, 3600, 29]    lable: [437, 3600, 30]

# [batch_size, sequence_len, 29] => [batch_size, sequence_len, 15, 29]
def input_prepare(inputs, splite_size):
    shape = np.array(inputs).shape
    batch_size, sequence_len, dims = shape[0], shape[1], shape[2]

    # padding = np.zeros((batch_size, splite_size-1, dims))
    # inputs = np.append(inputs, padding, axis=1)

    result = []
    for i in range(batch_size):
        temp_list = []
        for j in range(sequence_len-splite_size+1):
            temp_list.append(inputs[i][j: j+splite_size, :])
            shp = np.shape(inputs[i][j: j+splite_size, :])
            if shp[0]!=splite_size:print(i,j,shp)
        result.append(temp_list)

    # print(np.array(result).shape)
    return result

def output_handle():
    pass