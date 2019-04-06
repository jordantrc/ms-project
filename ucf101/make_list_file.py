from __future__ import print_function
import sys
import os
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

classes = get_class_list(CLASS_INDEX_FILE)

folder = sys.argv[1]
output_file = sys.argv[2]
file_list = os.listdir(folder)

fd = open(output_file, 'w')

for f in file_list:
    class_name = f.split('_')[1]
    class_index = str(classes.index(class_name))
    
    full_file_path = os.path.join(folder, f)
    line_string = full_file_path + " " + class_index
    fd.write(line_string +"\n")
    print(line_string)