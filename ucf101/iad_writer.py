import tensorflow as tf
import numpy as np

def make_sequence_example(data, file_properties):
    """creates the tfrecord example"""

    features = {
        'example_id':       tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_properties.uid])),
        'label':            tf.train.Feature(int64_list=tf.train.Int64List(value=[file_properties.label])),
        'network_depth':    tf.train.Feature(int64_list=tf.train.Int64List(value=[len(data)]))
    }
        


    for layer, am_layer in enumerate(data):
        features['num_rows/{:02d}'.format(layer + 1)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[am_layer.shape[0]]))# model spec
        features['num_columns/{:02d}'.format(layer + 1)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[am_layer.shape[1]]))# model spec
        features['img/{:02d}'.format(layer + 1)] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[am_layer.tostring()]))

        

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example

def write_tfrecord_to_disk(filename, data, info_values):

    # convert data to sequence example
    ex = make_sequence_example(data, info_values)

    # write sequence example to file
    writer = tf.python_io.TFRecordWriter(filename)
    writer.write(ex.SerializeToString())

def parse_sequence_example(filename_queue):
    '''
    Reads a TFRecord and separate the output into its constituent parts
        -filename_queue: a filename queue as generated using string_input_producer
    '''
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    
    # Read value features (data, labels, and identifiers)
    context_features = {        
        "example_id": tf.FixedLenFeature([], dtype=tf.string),
        "label": tf.FixedLenFeature([], dtype=tf.int64),
        "network_depth": tf.FixedLenFeature([], dtype=tf.int64)
    }
    
    # Read string features (input data)
    for i in range(1, 6):
        context_features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)
        context_features['num_rows/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        context_features['num_columns/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
    
    # Parse the example
    context_parsed = tf.parse_single_example(example,context_features)
    #return context_parsed
    # Decode the byte data of the data sequences
    sequence_data = {}
    for i in range(1, 6):
        sequence_data['img/{:02d}'.format(i)] = tf.decode_raw(context_parsed['img/{:02d}'.format(i)], np.float32)# float values for IAD
            
    return context_parsed, sequence_data
