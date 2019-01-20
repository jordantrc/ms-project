# iad_analysis.py

import csv
import numpy as np
import os

from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier

base_dir = ""
pca_n_components = list(range(2, 21))
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

    for sample_size in ['25', '50', '75', '100']:
        for layer in ['1', '2', '3', '4', '5']:
            train_file = "train_%s_%s.csv" % (sample_size, layer)
            test_file = "test_%s.csv" % [layer]

            train_path = os.path.join(base_dir, train_file)
            test_path = os.path.join(base_dir, test_file)

            # open the file and load the data
            train = []
            test = []

            with open(train_path, newline='') as csv_fd:
                csv_reader = csv.reader(csv_fd)
                for row in csv_reader:
                    train.append(row)

            with open(test_path, newline='') as csv_fd:
                csv_reader = csv.reader(csv_fd)
                for row in csv_reader:
                    test.append(row)

            train_x, train_y = list_to_numpy(train)
            test_x, test_y = list_to_numpy(test)

            # principal component analysis, cross-validation
            for n in pca_n_components:
                ipca = IncrementalPCA(n_components=n, batch_size=ipca_batch_size)
                ipca.fit(train_x)
                train_x_ipca = ipca.transform(train_x)
                test_x_ipca = ipca.transform(test_x)

                # classify with KNN and principcal components
                for k in knn_k:
                    classifier = KNeighborsClassifier(k)
                    classifier.fit(train_x_ipca, train_y)
                    knn_score = classifier.score(test_x_ipca, test_y)

                    print("%s, %s, %s, %.04f" % (sample_size, layer, n, knn_score))


if __name__ == "__main__":
    main()
