# tfrecord_to_csv.py
#
# converts a list of tfrecords to a
# CSV file containing the features
# for a dataset.

import csv
import os
import sys
import tensorflow as tf
import time

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

    # separate into list of filenames and classes
    filenames = []
    classes = []
    for l in lines:
        f, c = l.split()
        filenames.append(f)
        classes.append(c)

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


    with tf.Session() as sess:
        features = {}
        features['label'] = tf.FixedLenFeature((), tf.int64, default_value=0)
        for layer in range(1, 6):
            features['img/{:02d}'.format(layer)] = tf.FixedLenFeature((), tf.string)

        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        parsed_features = tf.parse_single_example(serialized_example, features)
        frames = []
        for layer in range(1, 6):
            frames.append(tf.decode_raw(parsed_features['img/{:02d}'.format(layer)], tf.float32))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        while True:
            start = time.time()
            try:
                frames_output = sess.run(frames)
                f = filenames[step]
                c = classes[step]
                f_base = os.path.basename(f)
                for i in range(5):
                    geom = LAYER_GEOMETRY[str(i + 1)]
                    row = [f_base, c]
                    frame_row = frames_output[i]
                    frame_row = list(frame_row)
                    assert len(frame_row) == geom[0] * geom[1], "layer = %s, len(frame_row) = %s, should be %s" % (layer, len(frame_row), geom[0] * geom[1])
                    row.extend(frame_row)
                    csv_writers[i].writerow(row)
                end = time.time()
                elapsed_time = end - start
                print("[%s] read data from %s - %.03f" % (step, f, elapsed_time))
                step += 1
            except tf.errors.OutOfRangeError:
                print("Completed processing tfrecord files")
                break 


if __name__ == "__main__":
    main()