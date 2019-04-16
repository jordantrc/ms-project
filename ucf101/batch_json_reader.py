import iad_writer_json
import random
import numpy as np
import os

from threading import Thread

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

    def get_batch_parallel(self):
        '''gets the data batch using multiple threads'''
        batch_data = [None] * self.batch_size
        batch_label = [None] * self.batch_size
        sample_names = [None] * self.batch_size

        while True:
            num_empty_slots = batch_data.count(None)
            if num_empty_slots == 0:
                break

            # dispatch threads
            if num_empty_slots < self.read_threads:
                num_threads_to_dispatch = num_empty_slots
            else:
                num_threads_to_dispatch = self.read_threads

            threads = [None] * num_threads_to_dispatch
            thread_data = [None] * num_threads_to_dispatch
            thread_label = [None] * num_threads_to_dispatch
            thread_name = [None] * num_threads_to_dispatch
            selected_files = []
            selected_positions = []

            # pick the files to read
            while len(selected_files) < num_threads_to_dispatch:
                if self.shuffle:
                    filename, start_pos, _ = random.choice(self.files).split()
                else:
                    filename, start_pos, _ = self.files[self.current_position].split()

                if filename not in selected_files:
                    selected_files.append(filename)
                    selected_positions.append(start_pos)
                    if not self.shuffle:
                        self.current_position = (self.current_position + 1) % len(self.files)

            # dispatch the threads
            for i in range(num_threads_to_dispatch):
                filename = selected_files[i]
                start_pos = selected_positions[i]

                threads[i] = Thread(target=self.thread_reader, args=(filename, start_pos, thread_data, thread_label, i))
                threads[i].start()

            for i in range(len(threads)):
                threads[i].join()

            # add the non-None results to the batch_data, batch_label lists
            for i, d in enumerate(thread_data):
                if d is not None:
                    first_empty_slot = batch_data.index(None)
                    batch_data[first_empty_slot] = d
                    batch_label[first_empty_slot] = thread_label[i]
                    sample_names[first_empty_slot] = os.path.basename(selected_files[i])

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)

        return batch_data, batch_label, sample_names

    def thread_reader(self, filename, start_pos, thread_data, thread_label, tid):
        '''thread-safe JSON file reader'''
        data, label = iad_writer_json.read_json_file_specific_depth(filename, self.c3d_depth)

        mod_start = int(int(start_pos) / self.divisor)

        data = data[:, mod_start:mod_start + self.window_size]
        #print("data shape = %s" % str(data.shape))
        #print("label = %s" % label)
        if data.shape[1] == self.window_size:
            thread_data[tid] = data
            thread_label[tid] = label
        print("thread %s done" % tid)

    
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
