import numpy as np
import os
import threading

from collections import namedtuple
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


TrainTestData_inst = namedtuple("train_test_data", "X_train y_train names_train indices_train "
                                                   "X_test y_test names_test indices_test")

def get_classes(inputdir, filext):
    dir = os.fsencode(inputdir)

    classes = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(filext):
            classes.append(filename)

    return classes


def get_rawdata(class_id, input_file, keep_filter):
    # for SIVAL data, first column is the image name (bag name).
    names = np.genfromtxt(input_file, delimiter=",", usecols=0, dtype='str')
    feat_data = np.genfromtxt(input_file, delimiter=",", dtype='object')[:, 1:]
    # Keep only those entries where the bag is positive (duplicated in other datasets)
    keep_filter = keep_filter.split(".")[0]
    keep = np.char.find(np.char.lower(names), keep_filter.lower()) > 0

    names = names[keep]
    names = names.reshape((len(names), 1))
    feat_data = feat_data[keep, :]
    classes = np.full_like(names, keep_filter).reshape((len(names), 1))
    # replace all positive instances' label with class id (categorical labeling)
    data = np.hstack([classes, names, feat_data])
    # Because I don't like bytestrings
    data[:, 2:] = data[:, 2:].astype('float32')
    data[data[:, -1] == 1, -1] = class_id + 1
    return data


def get_data_mt_wrapper(thread_id: int, dir, classes, dest_array):
    target_file = os.path.join(dir, classes[thread_id])
    data = get_rawdata(thread_id, target_file, classes[thread_id])
    dest_array[thread_id] = data
    return


def get_data(dir: str, extension: str, verbose=False):
    """

    :param dir: Search directory.
    :param extension: File extension to be parsed.
    :return: A numpy array of strings corresponding to the input data. Still needs to be trimmed/cast to an appropriate
    data type.
    """

    # get list of all files with extension in dir
    data_classes = get_classes(dir, extension)

    # initialise
    data = [None]*len(data_classes)
    threads = []

    for i in range(len(data_classes)):
        threads.append(threading.Thread(target=get_data_mt_wrapper, args=(i, dir, data_classes, data)))

    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()
        if verbose:
            print("Read data from " + data_classes[i])

    data = np.vstack(data)

    return data


def split_into_bags(input_array, bag_id_column):
    unique, s = np.unique(input_array[:, bag_id_column], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s)
    unique, s = unique[indices], s[indices]

    split_array = np.asarray(np.split(input_array, s[1:]), dtype='object')
    bag_labels = np.vstack([np.max(bag[:, -1]) for bag in split_array])

    # keep the arrays as flat as possible because nesting is confusing enough as it is
    return bag_labels, split_array


def split_inst(thread_id, dest_array, X, y, names, indices, omit_first_cols=0):
    indices_train = indices[0]
    indices_test = indices[1]

    # Split x
    X_train, X_test = X[indices_train], X[indices_test]
    names_train, names_test = names[indices_train], names[indices_test]
    y_train, y_test = np.array(y, dtype='int32')[indices_train], np.array(y, dtype='int32')[indices_test]

    # Now preprocess X
    X_train_pp = preprocessing.scale(X_train[:, omit_first_cols:])
    X_train = np.hstack([X_train[:, :omit_first_cols], X_train_pp])

    X_test_pp = preprocessing.scale(X_test[:, omit_first_cols:])
    X_test = np.hstack([X_test[:, :omit_first_cols], X_test_pp])

    modeldata = TrainTestData_inst(X_train=X_train, y_train=y_train, names_train=names_train, indices_train=indices_train,
                                   X_test=X_test, y_test=y_test, names_test=names_test, indices_test=indices_test)

    dest_array[thread_id] = modeldata
    return


def split_and_proc_instances(n_splits, data, omit_first_cols=0, verbose=False):
    # Split and divide into threads.

    # First three columns are labels, don't preprocess these/pass them along at all
    X = data[:, :-1]
    names = data[:, :1]
    y = data[:, -1:]
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=None)  # random state seems to trip up
    splits = sss.split(X, y)

    subthreads = [None] * n_splits
    dest_array = [None] * n_splits
    # need to loop through generator here as it is like Michelle in 'Allo 'Allo (only callable once, so you'd get race
    # conditions otherwise)
    for i, indices in enumerate(splits):
        subthreads[i] = threading.Thread(target=split_inst,
                                         args=(i, dest_array, X, y, names, indices, omit_first_cols))

    for i, thread in enumerate(subthreads):
        thread.start()
    for i, thread in enumerate(subthreads):
        thread.join()
    if verbose:
        print("Finished splitting data.")

    return dest_array
