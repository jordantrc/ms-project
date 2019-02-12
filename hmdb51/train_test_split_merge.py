# train_test_split_merge.py

import os

folder = "train-test-splits"
class_index_file = "train-test-splits/class_index.txt"

files = os.listdir(folder)
files.remove("class_index.txt")

test_list1 = []
test_list2 = []
test_list3 = []
train_list1 = []
train_list2 = []
train_list3 = []

for f in files:
    class_name = "_".join(f.split("_")[:-2])
    split_number_re = re.match(r'[a-z_]+(?P<split>\d).txt', f)
    split_number = split_number_re.group('split')

    # open the file and read the contents
    with open(f, 'r') as fd:
        raw_text = fd.read()
    lines = raw_text.split('\n')
    lines.remove('')

    if split_number == '1':
        
