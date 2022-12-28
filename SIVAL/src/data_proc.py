import os
import threading
from collections import namedtuple

import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import IterativeStratification

import synth_data

TrainTestData = namedtuple("train_test_data", "X_train y_train names_train indices_train "
                                              "X_test y_test names_test indices_test "
                                              "X_valid y_valid names_valid indices_valid")


def get_classes(inputdir, filext):
    folder = os.fsencode(inputdir)

    classes = []
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(filext):
            classes.append(filename)

    return classes


def get_rawdata(class_id, input_file, keep_filter, synthetic=False, model=None):
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

    if synthetic:
        data = synth_data.augment_dataset(data, model=model)

    return data


def get_data_mt_wrapper(thread_id: int, dir, classes, dest_array, synthetic=False, model=None):
    target_file = os.path.join(dir, classes[thread_id])
    data = get_rawdata(thread_id, target_file, classes[thread_id], synthetic, model)
    dest_array[thread_id] = data
    return


def get_data(folder: str, extension: str, synthetic=False, model=None, verbose=False):
    """

    :param folder: Search directory.
    :param extension: File extension to be parsed.
    :return: A numpy array of strings corresponding to the input data. Still needs to be trimmed/cast to an appropriate
    data type.
    """

    # get list of all files with extension in dir
    data_classes = get_classes(folder, extension)

    # initialise
    data = [None] * len(data_classes)
    threads = []

    if synthetic:
        print("Go grab a coffee, this will take a while")

    for i in range(len(data_classes)):
        threads.append(threading.Thread(target=get_data_mt_wrapper, args=(i, folder,
                                                                          data_classes, data,
                                                                          synthetic, model)))

    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()
        if verbose:
            print("Read data from " + data_classes[i])

    data = np.vstack(data)
    return data


def split_proc_inst(thread_id, dest_array, X, y, names, indices, validation_split=0.2, omit_first_cols=0,
                    omit_last_cols=0):
    indices_train = indices[0]
    indices_test = indices[1]

    # Split x
    X_train, X_test = X[indices_train], X[indices_test]
    names_train, names_test = names[indices_train], names[indices_test]
    y_train, y_test = np.array(y, dtype='int32')[indices_train], np.array(y, dtype='int32')[indices_test]

    # Create our own validation dataset as the split used by the model's .fit method only takes the last x%
    # ==> for a sorted list, the training dataset will miss some classes
    indices_valid, y_valid, names_valid = None, None, None

    if validation_split != 0:
        synth_labels = X_train[:, 2]
        y_train_flat = y_train.flatten()
        # multilabel_col = np.array([(synth_labels[i], y_train_flat[i]) for i, label in enumerate(y_train_flat)])  # or np.vstack.T?
        multilabel_col = np.array([(1, y_train_flat[i]) for i, label in enumerate(y_train_flat)])  # override

        k_fold = IterativeStratification(n_splits=int(1 / validation_split), order=2)
        splits = k_fold.split(X_train,
                              multilabel_col)  # Need to cast here or scipy complains about object dtype

        # Randomly select one of the splits (better way to do this?)
        stop_int = np.random.randint(0, int(1 / validation_split) - 1)
        for i, indices in enumerate(splits):
            if i == stop_int:
                break

        indices_train = indices[0]
        indices_valid = indices[1]  # Use later as we normalise/preprocess validation set together with training set

        names_valid = names_train[indices_valid]
        names_train = names_train[indices_train]

        y_valid = y_train[indices_valid]
        y_train = y_train[indices_train]

    # Now preprocess X
    if omit_last_cols == 0:
        X_train_pp = preprocessing.scale(X_train[:, omit_first_cols:])
        X_test_pp = preprocessing.scale(X_test[:, omit_first_cols:])
    else:
        X_train_pp = preprocessing.scale(X_train[:, omit_first_cols:-omit_last_cols])
        X_train_pp = np.hstack([X_train_pp, X_train[:, -omit_last_cols:]])
        X_test_pp = preprocessing.scale(X_test[:, omit_first_cols:-omit_last_cols])
        X_test_pp = np.hstack([X_test_pp, X_test[:, -omit_last_cols:]])

    X_train = np.hstack([X_train[:, :omit_first_cols], X_train_pp])
    X_valid = X_train[indices_valid]
    X_train = X_train[indices_train]
    X_test = np.hstack([X_test[:, :omit_first_cols], X_test_pp])

    modeldata = TrainTestData(X_train=X_train, y_train=y_train, names_train=names_train, indices_train=indices_train,
                              X_test=X_test, y_test=y_test, names_test=names_test, indices_test=indices_test,
                              X_valid=X_valid, y_valid=y_valid, names_valid=names_valid, indices_valid=indices_valid)

    dest_array[thread_id] = modeldata
    return


def split_proc_inst_wrapper(n_splits, data, validation_split=0.2, omit_first_cols=0, omit_last_cols=0, verbose=False):
    # First four columns are labels
    X = data[:, :-1]
    names = data[:, :1]
    synth_labels = data[:, 2]
    y = data[:, -1:]
    y_flat = y.flatten().astype('int32')

    # multilabel_col = np.array([(synth_labels[i], y_flat[i]) for i, label in enumerate(y_flat)])  # or np.vstack.T?
    multilabel_col = np.array([(1, y_flat[i]) for i, label in enumerate(y_flat)])  # quick and dirty override...

    sss = IterativeStratification(n_splits=n_splits, order=2)
    splits = sss.split(X, multilabel_col)

    subthreads = [None] * n_splits
    dest_array = [None] * n_splits
    # need to loop through generator here as it is like Michelle in 'Allo 'Allo (only callable once)
    for i, indices in enumerate(splits):
        subthreads[i] = threading.Thread(target=split_proc_inst,
                                         args=(i, dest_array, X, y, names, indices,
                                               validation_split, omit_first_cols, omit_last_cols))

    for i, thread in enumerate(subthreads):
        thread.start()
    for i, thread in enumerate(subthreads):
        thread.join()
    if verbose:
        print("Finished splitting instance data.")

    return dest_array


def split_proc_bag(thread_id, X_full, y_full, dest_array, indices, validation_split, omit_columns):
    indices_train = indices[0]
    indices_test = indices[1]

    X_train, y_train = \
        np.array(X_full, dtype='object')[indices_train], np.array(y_full, dtype='object')[indices_train]
    X_test, y_test = \
        np.array(X_full, dtype='object')[indices_test], np.array(y_full, dtype='object')[indices_test]

    # From X_train and y_train, create a validation dataset
    if validation_split != 0:
        synth_label = np.array([bag[0, 2] for bag in X_train])
        y_train_split = np.argmax(y_train, axis=1)
        # multilabel_col = np.array([(synth_label[i], y_train_split[i]) for i, label in enumerate(y_train_split)])
        multilabel_col = np.array([(1, y_train_split[i]) for i, label in enumerate(y_train_split)])  # quick and dirty override

        k_fold = IterativeStratification(n_splits=int(1 / validation_split), order=2)  # Somehow this isn't ok
        splits = k_fold.split(X_train,
                              multilabel_col)  # Need to cast here or scipy complains about object dtype

        # Randomly select one of the splits (better way to do this?)
        stop_int = np.random.randint(0, int(1 / validation_split) - 1)
        for i, indices in enumerate(splits):
            if i == stop_int:
                break

        indices_train = indices[0]
        indices_valid = indices[1]  # Use later as we normalise/preprocess validation set together with training set

    # Preprocessing
    # This seems convoluted but no immediately obvious way to do this that doesn't involve overwriting the scale func...
    # Flatten, normalise and then split sets again
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    unique_train, s_train = np.unique(X_train[:, 1], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s_train)
    unique_train, s_train = unique_train[indices], s_train[indices]

    unique_test, s_test = np.unique(X_test[:, 1], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s_test)
    unique_test, s_test = unique_test[indices], s_test[indices]

    names_train = X_train[s_train[indices_train]][:, 1]
    names_valid = (X_train[s_train[indices_valid]][:, 1] if validation_split == 0 else None)
    names_test = X_test[s_test][:, 1]

    # Scale and then re-append the info of the whole shebang... Mostly to have the instance info and label
    X_train = np.hstack((X_train[:, 0:omit_columns], preprocessing.scale(X_train[:, omit_columns:-1]), X_train[:, -1:]))
    X_train = np.split(X_train, s_train[1:])  # re-bagged, normalised data, from which we select training and validation
    X_train = np.asarray(X_train, dtype='object')

    X_valid = (X_train[indices_valid] if validation_split != 0 else None)
    y_valid = (y_train[indices_valid] if validation_split != 0 else None)

    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    X_test = np.hstack((X_test[:, 0:omit_columns], preprocessing.scale(X_test[:, omit_columns:-1]), X_test[:, -1:]))
    X_test = np.split(X_test, s_test[1:])

    modeldata = TrainTestData(X_train=X_train, y_train=y_train, names_train=names_train, indices_train=indices_train,
                              X_test=X_test, y_test=y_test, names_test=names_test, indices_test=indices_test,
                              X_valid=X_valid, y_valid=y_valid, names_valid=names_valid, indices_valid=indices_valid)

    dest_array[thread_id] = modeldata
    return


def split_proc_bag_wrapper(no_splits, input_data, data_labels, validation_split, omit_first_cols, verbose=False):
    X = input_data  # drop img class and instance label later
    # below is not needed if you use SparseCategoricalAccuracy
    data_labels = data_labels.flatten().astype('int32')
    y = to_categorical(data_labels)  # for iterative stratification
    synth_label = np.array([bag[0, 2] for bag in input_data])
    # multilabel_col = np.array([(synth_label[i], data_labels[i]) for i, label in enumerate(data_labels)])  # or np.vstack.T?
    multilabel_col = np.array([(1, data_labels[i]) for i, label in enumerate(data_labels)])  # quick and dirty override

    # http://scikit.ml/api/skmultilearn.model_selection.iterative_stratificneation.html
    k_fold = IterativeStratification(n_splits=no_splits, order=2)
    splits = k_fold.split(X, multilabel_col)

    # Divide into threads
    subthreads = [None] * no_splits
    dest_array = [None] * no_splits
    # need to loop through generator here as it is like Michelle in 'Allo 'Allo (only callable once, so you'd get race
    # conditions otherwise)
    for i, indices in enumerate(splits):
        subthreads[i] = threading.Thread(target=split_proc_bag,
                                         args=(i, X, y, dest_array, indices, validation_split, omit_first_cols))

    for i, thread in enumerate(subthreads):
        thread.start()
    for i, thread in enumerate(subthreads):
        thread.join()

    if verbose:
        print("Finished setting up " + str(no_splits) + " bag training/test data sets")

    return dest_array


def preprocess_bagged_data(input_data, omit_first_cols):
    # Don't forget to preprocess. So flatten, normalise, resplit
    data_flattened = np.vstack(input_data)
    data_flattened_proc = np.hstack([data_flattened[:, :omit_first_cols],
                                     preprocessing.scale(data_flattened[:, omit_first_cols:-1]),
                                     data_flattened[:, -1:]])

    unique, s = np.unique(data_flattened[:, 1], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s)
    unique, s = unique[indices], s[indices]

    data_split_proc = np.split(data_flattened_proc, s[1:])

    return data_split_proc


def append_bag_labels(data, predicted_bag_labels, predicted_bag):
    labels = [None] * data.shape[0]
    for j, instance in enumerate(data):
        inst_name = instance[1]
        index = predicted_bag_labels.index(inst_name)
        inst_class = predicted_bag[index]
        labels[j] = inst_class

    labels = np.reshape(labels, (data.shape[0], 1))
    labeled_dataset = np.hstack([data, labels])

    return labeled_dataset
