# iad_analysis.py

import csv
import numpy as np
import os
import sys

from sklearn.decomposition import IncrementalPCA

base_dir = "/home/jordanc/datasets/UCF-101/csv/"
pca_n_components = list(range(2, 101))
ipca_batch_size = 1000
knn_k = list(range(3, 20))


def list_to_numpy(data):
    '''create a numpy array of the features'''
    x = np.zeros([len(data) - 1, len(data[0]) - 2])
    y = np.zeros([len(data) - 1], np.uint8)

    i = 0
    for t in data[1:]:
        j = 0
        for f in t[2:]:
            x[i][j] = float(f)
            j += 1
        y[i] = int(t[1])
        i += 1

    return x, y


def main():
    '''main function'''

    for sample_size in ['50', '75', '100']:
        layer = '3'
        train_file = "train_%s_%s.csv" % (sample_size, layer)
        test_file = "test_%s.csv" % (layer)

        train_path = os.path.join(base_dir, train_file)
        test_path = os.path.join(base_dir, test_file)

        # open the file and load the data
        train = []
        test = []

        with open(train_path, 'rb') as csv_fd:
            csv_reader = csv.reader(csv_fd)
            for row in csv_reader:
                train.append(row)

        with open(test_path, 'rb') as csv_fd:
            csv_reader = csv.reader(csv_fd)
            for row in csv_reader:
                test.append(row)

        train_x, train_y = list_to_numpy(train)
        test_x, test_y = list_to_numpy(test)

        # principal component analysis, cross-validation
        for n in pca_n_components:
            ipca = IncrementalPCA(n_components=n, batch_size=ipca_batch_size)
            ipca.fit(train_x)

            print("ipca_components shape = %s" % str(ipca.explained_variance_ratio_.shape))
            explained_variance = [str(x) for x in ipca.explained_variance_ratio_.tolist()]
            explained_variance_str = ",".join(explained_variance)

            print("explained_variance,%s,%s,%s,%s" % (sample_size, layer, n, explained_variance_str))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
