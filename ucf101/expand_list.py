# expand_list.py
#
# Expands a test or training list based
# on the contents of a directory. Finds all sub-samples
# belonging to each sample in the list.
#
# Usage: expand_list.py <list file> <directory> <extension> <output file>
#

import os
import sys

def main():
    '''main function'''
    if len(sys.argv) != 5:
        print("invalid number of arguments [%s]" % len(sys.argv))
        sys.exit(1)

    list_file = sys.argv[1]
    directory = sys.argv[2]
    assert os.path.isdir(directory)
    extension = sys.argv[3]
    output_file = sys.argv[4]

    # get the samples from the file
    with open(list_file, 'r') as fd:
        text = fd.read()
        lines = text.split('\n')
        while '' in lines:
            lines.remove('')

    # list the directory, filter based on extension
    dir_list = os.listdir(directory)
    dir_list = [x for x in dir_list if extension in x]

    # traverse the list of samples, looking for subsamples
    fd = open(output_file, 'w')
    subsamples = []
    for f in sorted(dir_list):
        base = "_".join(f.split('_')[0:4])
        for l in lines:
            if base in l:
                s, c = l.split()
                sample_path = os.path.join(directory, f)
                fd.write("%s %s\n" % (sample_path, c))

    fd.close()


if __name__ == "__main__":
    main()
