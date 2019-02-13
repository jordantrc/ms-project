# train_test_split_merge.py

import os
import re

folder = "train-test-splits"
class_index_file = "train-test-splits/class_index.txt"

# get the class indicies
class_indices = {}
with open(class_index_file, 'r') as fd:
    raw_text = fd.read()
lines = raw_text.split('\n')
lines.remove('')
for l in lines:
    class_name, index = l.split()
    class_indices[class_name] = int(index)

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

    if split_number == '1':
        modify_train = train_list1
        modify_test = test_list1
    elif split_number == '2':
        modify_train = train_list2
        modify_test = test_list2
    elif split_number == '3':
        modify_train = train_list3
        modify_test = test_list3

    # open the file and read the contents
    file_path = os.path.join(folder, f)
    with open(file_path, 'r') as fd:
        raw_text = fd.read()
    lines = raw_text.split('\n')
    lines.remove('')

    for l in lines:
        training_video = False
        testing_video = False
        file_name, inclusion_digit = l.split()
        if inclusion_digit == '0':
            continue
        elif inclusion_digit == '1':
            training_video = True
        elif inclusion_digit == '2':
            testing_video = True

        # determine the class index
        class_index = class_indices[class_name]

        # now determine which list to include the video in
        if training_video:
            modify_train.append([class_name + "/" + file_name, str(class_index)])
        elif testing_video:
            modify_test.append([class_name + "/" + file_name, str(class_index)])

# write the training and testing files
with open("trainlist01.txt", 'w') as fd:
    for t in train_list1:
        fd.write(" ".join(t) + "\n")

with open("testlist01.txt", 'w') as fd:
    for t in test_list1:
        fd.write(" ".join(t) + "\n")

with open("trainlist02.txt", 'w') as fd:
    for t in train_list2:
        fd.write(" ".join(t) + "\n")

with open("testlist02.txt", 'w') as fd:
    for t in test_list2:
        fd.write(" ".join(t) + "\n")

with open("trainlist03.txt", 'w') as fd:
    for t in train_list3:
        fd.write(" ".join(t) + "\n")

with open("testlist03.txt", 'w') as fd:
    for t in test_list3:
        fd.write(" ".join(t) + "\n")
