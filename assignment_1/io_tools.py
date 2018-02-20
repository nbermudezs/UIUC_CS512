import numpy as np


def read_dataset(filename, data_folder):
    with open(data_folder + '/' + filename, 'r') as f:
        x = []
        y = []

        for line in f:
            instance = line.split(' ')
            y.append(instance[0])
            x.append([float(x) for x in instance[1:-1]])

        return np.array(y), np.array(x)


def read_phrase_list(filename, data_folder):
    with open(data_folder + '/' + filename, 'r') as f:
        conf = []
        phrases = []

        for line in f:
            instance = line.split('\t')
            conf.append(float(instance[0]))
            phrases.append(instance[1])

        return np.array(conf), np.array(phrases)


def save_cluster_numpy(X, y, data_folder):
    # np.save(data_folder + '/' + 'cluster-X.np', X)
    np.save(data_folder + '/' + 'cluster-y.np', y)


def load_cluster_numpy(data_folder):
    return np.load(data_folder + '/' + 'cluster-y.np')