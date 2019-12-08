import pickle
import subprocess
from scipy import spatial

pkl_dataset_path = "img_dataset_facenet.pkl"
pkl_input_path = "img_facenet.pkl"

# make input pkl
res = subprocess.call('python src/feature_extract.py ./model/20180408-102900 ./data/images/*')

# read dataset
with open(pkl_dataset_path, 'rb') as f:
    data = pickle.load(f)

key_list = []
for key in data.keys():
    key_list.append(key)

# read input_data
with open(pkl_input_path, 'rb') as f:
    input_data = pickle.load(f)

key_input_list = []
for key in input_data.keys():
    key_input_list.append(key)

# compare distance
for input_key in key_input_list:
    output_name = ""
    min_distance = 1000000000
    for key in key_list:
        current_data = data[key]
        assert(len(key_input_list))
        check_data = input_data[input_key]
        distance = spatial.distance.euclidean(current_data, check_data)
        if(min_distance > distance):
            min_distance = distance
            output_name = key
    
    print("input_data is " + input_key)
    print(" -> answer is " + output_name[:-4])

# print(spatial.distance.euclidean(A, B))