import numpy as np
from collections import defaultdict


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


def read_segmentation_metrics(filename, data_folder):
    all_tuples = defaultdict(lambda: [])
    with open(data_folder + '/' + filename) as f:
        for line in f:
            if not line.startswith('Performing phrasal_segmentation.sh'):
                continue

            pieces = line.split()
            key = pieces[2].split('/')[-1]
            multi = float([x for x in pieces if
                           x.startswith('HIGHLIGHT_MULTI')][0].split('=')[1])
            single = float([x for x in pieces if
                            x.startswith('HIGHLIGHT_SINGLE')][0].split('=')[1])

            while True:
                if 'Phrasal segmentation finished' in next(f):
                    break
            total_phrases = int(next(f).split()[-1])
            next(f)
            sentence_avg = float(next(f).split()[-1])
            # print(single, multi, total_phrases, sentence_avg)
            all_tuples[key].append((single, multi, total_phrases, sentence_avg))
    return all_tuples
