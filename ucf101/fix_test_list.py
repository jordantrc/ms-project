
import sys
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

classes = get_class_list(CLASS_INDEX_FILE)

filename = sys.argv[1]

with open(filename) as fd:
	text = fd.read()
	lines = text.split('\n')

with open(filename, 'w') as fd:
	for l in lines:
		if len(l) > 0:
			directory, file = l.split('/')
			label = file.split('_')[1]
			int_label = classes.index(label)
			print("directory = %s, file = %s, label = %s" % (directory, file, str(int_label)))
			fd.write(directory + "/" + file + " " + str(int_label) + "\n")
