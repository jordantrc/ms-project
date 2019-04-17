import iad_writer_json
import random
import numpy as np
import os

import multiprocessing
import time
from multiprocessing import Pool

def convert_files_to_batchable_format(filename, directory, outfilename):
    all_files = os.listdir(directory)
    all_files.sort()
    
    ifile = list(open(filename, 'r'))

    txt = ''
    for line in ifile:
        filename, start_pos, label = line.split()

        mod_filename = filename.split('/')[-1]+".json"
        if(mod_filename in all_files):
            txt += os.path.join(directory, mod_filename) + ' ' + start_pos + ' ' + label + '\n'
    
    ofile = open(outfilename, 'w')
    ofile.write(txt)
    ofile.close


def get_data(values):
    filename, start_pos, c3d_depth = values[0], values[1], values[2]
    data, label = iad_writer_json.read_json_file_specific_depth(filename, c3d_depth)
        
    divisor = [1,1,2,4,8][c3d_depth]
    window_size = [16,16,8,4,2][c3d_depth]

    mod_start = int(start_pos)/divisor

    data = data[:, mod_start:mod_start + window_size]

    if data.shape[1] == window_size:
        return (data, label, os.path.basename(filename))
    else:
        return (None, None, None)


class BatchJsonRead():
    def __init__(self, filename, batch_size, c3d_depth, shuffle, read_threads=0):
        self.files = list(open(filename,'r'))
        self.current_position = 0

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.c3d_depth = c3d_depth
        self.window_size = [16,16,8,4,2][c3d_depth]
        self.divisor = [1,1,2,4,8][c3d_depth]
        self.discards = 0
        self.read_threads = read_threads
        self.num_files = len(self.files)

        print("BatchJsonRead initialization complete, found %s JSON files" % len(self.files))

    def get_batch(self):
        '''calls the correct get_batch_* function based on the
        self.read_threads value'''
        if self.read_threads > 1:
            batch_data, batch_label, sample_names = self.get_batch_parallel()
        else:
            batch_data, batch_label, sample_names = self.get_batch()

        return batch_data, batch_label, sample_names

    def get_patch_parallel(self):
        '''parallel get_batch implementation'''
        batch_data = []
        batch_label = []
        sample_names = []
        
        while True:
            if len(batch_data) < self.read_threads:
                num_threads = self.read_threads - len(batch_data)
            else:
                num_threads = self.read_threads
            p = Pool(num_threads)
            values = []
            
            for i in range(num_threads):
                if self.shuffle:
                    filename, start_pos, _ = random.choice(self.files).split()
                else:
                    filename, start_pos, _ = self.files[self.current_position].split()
                    self.current_position = (self.current_position + 1) % len(self.files)
                values.append((filename, start_pos, self.c3d_depth))

            out = p.map(get_data, values)

            for o in out:
                if o[0] is not None:
                    batch_data.append(o[0])
                    batch_label.append(o[1])
                    sample_names.append(o[2])

                    if len(batch_data) == self.batch_size:
                        break
                else:
                    self.discards += 1

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)

        return batch_data, batch_label, sample_names
  
    def get_batch(self):
        batch_data = []
        batch_label = []
        sample_names = []

        while True:
            if(self.shuffle):
                filename, start_pos, _ = random.choice(self.files).split()
            else:
                filename, start_pos, _ = self.files[self.current_position].split()
                self.current_position = (self.current_position+1) % len(self.files)

            data, label = iad_writer_json.read_json_file_specific_depth(filename, self.c3d_depth)

            mod_start = int(int(start_pos)/self.divisor)

            data = data[:, mod_start:mod_start+self.window_size]
            #print("data shape = %s" % str(data.shape))
            #print("label = %s" % label)
            if data.shape[1] == self.window_size:
                # add the sample if the temporal shape is correct, otherwise, discard
                batch_data.append(data)
                batch_label.append(label)
                sample_names.append(os.path.basename(filename))
                #print("len(batch_data) = %s, batch_size = %s" % (len(batch_data), self.batch_size))
                if len(batch_data) == self.batch_size:
                    break
            else:
                #print("\tdiscarding %s, start %s" % (os.path.basename(filename), start_pos))
                self.discards += 1

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)

        return batch_data, batch_label, sample_names

if __name__ == '__main__':
    convert_files_to_batchable_format('train-test-splits/c3d_ucf101_train_split1.txt', 
                                      'D:/ms-project-data/UCF101/json_iads/generated_iads_train_100',
                                      'train-test-splits/ucf101train-hyperion.list')
    convert_files_to_batchable_format('train-test-splits/c3d_ucf101_test_split1.txt',
                                      'D:/ms-project-data/UCF101/json_iads/generated_iads_test_100',
                                      'train-test-splits/ucf101test-hyperion.list')

    
    reader = BatchJsonRead('train-test-splits/ucf101train-hyperion.list', 30, 0, True)

    data, label = reader.get_batch()
    for i in range(len(data)):
        print(i, data[i].shape, label[i])

    reader = BatchJsonRead('train-test-splits/ucf101test-hyperion.list', 30, 0, False)

    data, label = reader.get_batch()
    for i in range(len(data)):
        print(i, data[i].shape, label[i])
