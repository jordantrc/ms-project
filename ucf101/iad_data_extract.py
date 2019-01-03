# iad_data_extract.py

import tensorflow as tf
from tfrecord_gen import CLASS_INDEX_FILE, get_class_list

LAYER = 1
LAYER_GEOMETRY = {'1': (64, 16, 1),
                  '2': (128, 16, 1),
                  '3': (256, 8, 1),
                  '4': (512, 4, 1),
                  '5': (512, 2, 1)
                  }

def _parse_function(example):
    img_geom = tuple([1]) + LAYER_GEOMETRY[str(LAYER)]
    features = dict()
    features['label'] = tf.FixedLenFeature((), tf.int64)

    for i in range(1, 6):
        # features['length/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.int64)
        features['img/{:02d}'.format(i)] = tf.FixedLenFeature((), tf.string)

    parsed_features = tf.parse_single_example(example, features)

    # decode the image, get label
    img = tf.decode_raw(parsed_features['img/{:02d}'.format(LAYER)], tf.float32)
    img = tf.reshape(img, img_geom, "parse_reshape")

    #img = tf.image.resize_bilinear(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    print("img shape = %s" % img.get_shape())
    img = tf.squeeze(img, 0)

    label = tf.cast(parsed_features['label'], tf.int64)
    label = tf.one_hot(label, depth=NUM_CLASSES)

    return img, label


def main():

    filenames = []
    list_file = sys.argv[1]
    

    with open(list_file) as fd:
        lines = fd.read().split('\n')
        while '' in lines:
            lines.remove('')
        
        for l in lines:
            f, c = l.split()
            filenames.append(f)

    class_list = get_class_list(CLASS_INDEX_FILE)

    # dataset iterator
    input_filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    dataset_iterator = dataset.make_initializable_iterator()
    x, y_true = dataset_iterator.get_next()
    y_true_class = tf.argmax(y_true, axis=1)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(dataset_iterator.initializer, feed_dict={input_filenames: filenames})

        while True:
            try:
                x_out, y_true_out = sess.run([x, y_true])
            except tf.errors.OutOfRangeError:
                break

