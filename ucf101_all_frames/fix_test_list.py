from __future__ import print_function
import sys
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

classes = get_class_list(CLASS_INDEX_FILE)

filename = sys.argv[1]
fd = open(filename)
text = fd.read()
lines = text.split('\n')

with open(filename + "fixed", 'w+') as fd:
	for l in lines:
		if len(l) > 0:
			label, folder = l.strip().split('/')
			# print("%s/%s" % (label, folder))
			class_name = str(classes.index(label))
			string = label + '\\' + folder + ' ' + class_name
			print(string)
			fd.write(string + "\n")