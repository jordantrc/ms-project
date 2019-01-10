# tfrecord_to_csv.py
#
# converts a list of tfrecords to a
# CSV file containing the features
# for a dataset.

import csv
import os
import sys
import tensorflow as tf

from iad_cnn import LAYER_GEOMETRY


def main():
    '''main function'''
    list_file = sys.argv[1]
    output_file_base = sys.argv[2]

    # get the list of files from the list_file
    with open(list_file, 'r') as fd:
        raw = fd.read()
    lines = raw.split('\n')
    lines.remove('')

    # open the output files for writing, write the header
    csv_fds = []
    csv_writers = []
    for i in range(0, 5):
        header = ['sample_name', 'class']
        csv_fd = open(output_file_base + '_' + str(i) + '.csv', 'wb')
        csv_fds.append(csv_fd)
        csv_writer = csv.writer(csv_fd, dialect='excel')
        
        # construct the header
        geometry = LAYER_GEOMETRY[str(i + 1)]
        num_features = geometry[0] * geometry[1]
        feature_header = [str(x) for x in range(num_features)]
        header.extend(feature_header)

        # write the header
        csv_writer.writerow(header)
        csv_writers.append(csv_writer)

    # open each file in the lines list, parse the sample
    sess = tf.Session()
    for l in lines:
        f, c = l.split()
        f_base = os.path.basename(f)
        row = [f_base, c]
        for example in tf.python_io.tf_record_iterator(f):
            print("extracting data from %s" % (f))
            features = dict()
            features['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)

            for layer in range(1, 6):
                features['img/{:02d}'.format(layer)] = tf.FixedLenFeature((), tf.string)

            parsed_features = tf.parse_single_example(example, features)

            with sess.as_default():
                for layer in range(1, 6):
                    geom = LAYER_GEOMETRY[str(layer)]
                    frame = tf.decode_raw(parsed_features['img/{:02d}'.format(layer)], tf.float32)
                    frame = frame.eval()
                    frame_row = list(frame)
                    assert len(frame_row) == geom[0] * geom[1], "layer = %s, len(frame_row) = %s, should be %s" % (layer, len(frame_row), geom[0] * geom[1])
                    row.extend(frame_row)
                    csv_writers[layer - 1].writerow(row)  


if __name__ == "__main__":
    main()