import tensorflow as tf
import numpy as np

import json

def write_json_to_disk(filename, data, info_values):
    '''
    save current un-thresholded data to file.
    '''
    json_dict = {"example_id": filename.split('/')[-1],
            "label": info_values.label,
            "depth": len(data),

            "num_rows": [],
            "num_columns": [],
            "data": []}

    for c3d_depth in range(len(data)):
        json_dict["num_rows"].append(data[c3d_depth].shape[0])
        json_dict["num_columns"].append(data[c3d_depth].shape[1])
        json_dict["data"].append(np.matrix(data[c3d_depth]).flatten().tolist()[0])

    json.dump(json_dict, open(filename, 'w'))

def read_json_file_entire(json_filename):
    with open(json_filename) as json_file:
        return json.load(json_file)

def read_json_file_specific_depth(json_filename, c3d_depth):
    with open(json_filename) as json_file:
        json_dict = json.load(json_file)

        data = np.array(json_dict["data"][c3d_depth]).reshape(json_dict["num_rows"][c3d_depth], json_dict["num_columns"][c3d_depth])
        label = json_dict["label"]

        return data, label