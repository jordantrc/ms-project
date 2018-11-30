from __future__ import print_function
import sys
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

classes = get_class_list(CLASS_INDEX_FILE)

filename = sys.argv[1]

with open(filename) as fd:
	text = fd.read()
	lines = text.split('\n')

#with open(filename, 'w+') as fd:
for l in lines:
	print("line = %s" % l)