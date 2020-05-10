file1 = '/Users/lr/Desktop/VMR/data/Analyze/score_list_0_4000_data2.pickle'
file2 = '/Users/lr/Desktop/VMR/data/Analyze/score_list_4000_8000_data2.pickle'
import pickle
import numpy as np

with open(file1, 'rb') as reader:
    data2_1 = np.array(pickle.load(reader))

with open(file2, 'rb') as reader:
    data2_2 = np.array(pickle.load(reader))

print(data2_1.shape)