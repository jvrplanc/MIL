import copy
import numpy as np
import os
#import pickle
import shelve
import sys
import tensorflow as tf
import threading

from collections import namedtuple
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

lock = threading.Lock()

TrainTestData = namedtuple("train_test_data", "X_train y_train names_train indices_train "
                                              "X_test y_test names_test indices_test")
ThreadInfo = namedtuple("thread_info", "class_no no_splits")

def get_classes(inputdir, filext):
    dir = os.fsencode(inputdir)

    classes = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(filext):
            #classes.insert(0, filename)
            classes.append(filename)

    return classes


def get_rawdata(inputfile):
    # for SIVAL data, first column is the image name (bag name).
    names = np.genfromtxt(inputfile, delimiter=",", usecols=0, dtype=str)
    data = np.genfromtxt(inputfile, delimiter=",", dtype=float)[:, 1:]
    return names, data


def get_data_wrapper(thread_id, classes, data):
    data_class = classes[thread_id]
    data_class_path = os.path.join(SIVAL_data_dir, data_class)
    data[thread_id][0] = data_class.split(".")[0]
    data[thread_id][1] = get_rawdata(data_class_path)
    print("Read data from " + alldata[thread_id][0])


def split_and_preprocess(thread_id, class_info: ThreadInfo, data, names, splits):
    subthread_id = thread_id + class_info.class_no * class_info.no_splits

    print("Starting subthread " + str(subthread_id) + ", returning data for class " + str(class_info.class_no))

    # no check for thread id <= sss.get_n_splits()!
    train_index, test_index = splits

    data_train, data_test = data[train_index], data[test_index]
    names_train, names_test = names[train_index], names[test_index]

    X_train = preprocessing.scale(data_train[:, 1:-1])
    y_train = data_train[:, -1]
    X_test = preprocessing.scale(data_test[:, 1:-1])
    y_test = data_test[:, -1]

    modeldata = TrainTestData(X_train=X_train, y_train=y_train, names_train=names_train, indices_train=train_index,
                              X_test=X_test, y_test=y_test, names_test=names_test, indices_test=test_index)

    train_test_data[subthread_id] = modeldata
    return


def nn_model_inst(nfeat):
    model = tf.keras.models.Sequential([  # as per Wang et al., but without dropout
        # to do: add kernel regularizers
        tf.keras.layers.Dense(units=256, activation='ReLU', input_dim=nfeat),
        tf.keras.layers.Dense(units=128, activation='ReLU'),
        tf.keras.layers.Dense(units=64, activation='ReLU'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')  # ,
        # tf.keras.layers.MaxPooling1D(pool_size=2)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model


def process_data_wrapper(thread_id, no_splits, data):
    print("Starting thread to generate dataset for class " + str(thread_id))

    # From data, extract X and y. First col is the instance id, last the instance label, rest contains feature vectors
    # Really need to be more consistent with array shapes and sizes, the one times it's a 1D array of n_splits*len(data),
    # the next it's a 2D array with shape n_splits, len(data). Error-prone!
    class_id = thread_id
    X = data[class_id][1][1][:, 1:-1]
    y = data[class_id][1][1][:, -1]

    # Split and divide into threads
    # use 25% of dataset because we run into memory errors otherwise (at least locally)
    sss = StratifiedShuffleSplit(n_splits=no_splits, test_size=0.1, random_state=None)  # random state seems to trip up
    splits = sss.split(X, y)

    thread_info = ThreadInfo(class_no=thread_id, no_splits=sss.get_n_splits())

    subthreads = [None]*no_splits
    # need to loop through generator here as it is like Michelle in 'Allo 'Allo (only callable once, so you'd get race
    # conditions otherwise)
    for i, indices in enumerate(splits):
        subthreads[i] = threading.Thread(target=split_and_preprocess,
                                         args=(i, thread_info, data[class_id][1][1], data[class_id][1][0], indices))

    for i, thread in enumerate(subthreads):
        thread.start()
    for i, thread in enumerate(subthreads):
        thread.join()

    print("Finished running data extraction code for image class " + str(class_id))
    return


def fit_and_test(thread_id, no_splits, mi_model, data: TrainTestData, no_epochs):
    class_id = thread_id//no_splits
    split_no = thread_id % no_splits
    batch_size = 20  # data.X_train//10 # this tripped up model.fit() for some as yet unknown reason

    # recompile model as they're all clones
    mi_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])  # Categorical probably makes more sense

    history[class_id][split_no] = \
        mi_model.fit(data.X_train, data.y_train, epochs=no_epochs, validation_split=0.2, batch_size=batch_size)

    threshold = .5
    if thread_id==10:
        pass
    predictions = (mi_model.predict(data.X_test, batch_size=batch_size) > threshold).astype(int)

    evals = model.evaluate(data.X_test, data.y_test, batch_size=batch_size)
    # print('Loss: {}'.format(evals[0]))
    # print('Accuracy: {}'.format(evals[1]))
    evalall[class_id][split_no] = evals

    idx0 = (data.y_test == 0)
    idx1 = (data.y_test == 1)

    # this still gives poor results for the non-first dataset and using 15 epochs (on acc1), but bag level stats are ok
    acc0 = np.mean(np.equal(predictions[idx0], data.y_test[idx0]))
    acc0all[class_id][split_no] = acc0
    acc1 = np.mean(np.equal(predictions[idx1], data.y_test[idx1]))
    acc1all[class_id][split_no] = acc1

    return


def test_bag_level(thread_id, no_splits, data, model_history, threshold = 0.5):
    # Here, the "validation" set is the entire .data file. M.o.:
    # 1. Verification dataset: 'flatten' the input dataset such that there is one entry per image. Label = 1 if positive
    # 2. For use with the actual model, run the unflattened dataset through the model and then apply the same operation.
    # 3. Compare
    class_id = thread_id//no_splits
    split_no = thread_id % no_splits
    dataset_name = data[0]

    # We need to do some preprocessing on the split. np.unique alphabetically sorts the list first, leading to a
    # non-sequential list of indices. We thus see what we'd need to do to sort this list of indices (i.e. something we
    # can pass to np.split(), and sort all concomitant arrays in the same way using argsort's output.
    unique, s = np.unique(data[1][0], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s)
    s = s[indices]  # sorted list of indices indicating start of new image.

    if thread_id == 10:
        pass
    names, data_grouped = np.split(data[1][0], s[1:]), np.split(data[1][1], s[1:])
    bag_data = [(names[i][0], np.max(data_grouped[i][:, -1])) for i in range(len(unique))]
    bag_gt = np.asarray(bag_data)[:, -1].astype(float).astype(int)

    # Run the whole dataset through the model. Discard instance id and label of course...
    instance_data = data[1][1][:, 1:-1]
    # Does preprocessing lead to data leakage in this case?
    instance_data = preprocessing.scale(instance_data)

    local_model = model_history.model
    # the instance predictions are only somewhat accurate for the first dataset (regardless of the model!!)
    # acc1 is 0 for all other datasets, so bag accuracy will also be rubbish.
    # i.e. the model for the second dataset is somehow training on the first?
    inst_predictions = (local_model.predict(instance_data) > threshold).astype(int)
    inst_predictions_grouped = np.split(inst_predictions, s[1:])
    bag_predictions = np.asarray([np.max(inst_predictions_grouped[i]) for i in range(len(unique))])

    idx0 = (bag_gt == 0)
    idx1 = (bag_gt == 1)

    acc0 = np.mean(np.equal(bag_predictions[idx0], bag_gt[idx0]))
    acc0all_bag[class_id][split_no] = acc0
    if thread_id == 10:
        pass
    acc1 = np.mean(np.equal(bag_predictions[idx1], bag_gt[idx1]))
    acc1all_bag[class_id][split_no] = acc1

    return

SIVAL_data_dir = "../input/amil-sival-debug/"
data_classes = get_classes(SIVAL_data_dir, ".data")

# new 2 column np array, col 0 is data classes and col 1 the data. Not sure whether this is the most efficient way
alldata = [[0 for i in range(2)] for j in range(len(data_classes))]

# 25 threads for 25 image classes, get these in parallel
threads = [threading.Thread(target=get_data_wrapper, args=(i, data_classes, alldata)) for i in range(len(data_classes))]

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()

# for each image class, generate 10 datasets
n_splits = 10

# pass all data so we can also select negative instances from other image classes
train_test_data = [None] * n_splits * len(data_classes)
threads = [threading.Thread(target=process_data_wrapper, args=(i, n_splits, alldata)) for i in range(len(data_classes))]

n = 0
for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()

# Not sure whether it makes sense to start and restart so many parallel threads vs just keeping it all in 1 thread
# per test/training split...
# We now have n_splits*no_classes datasets (250), all preprocessing.scaled.
model = nn_model_inst(nfeat=train_test_data[0][0].shape[1])
epochs = 25

history = [[None]*n_splits for i in range(len(data_classes))]
evalall = [[None]*n_splits for i in range(len(data_classes))]
acc0all = [[None]*n_splits for i in range(len(data_classes))]
acc1all = [[None]*n_splits for i in range(len(data_classes))]

threads = [threading.Thread(target=fit_and_test, args=(i, n_splits, tf.keras.models.clone_model(model),
                                                       train_test_data[i], epochs))
           for i in range(len(train_test_data))]

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()
for i in range(len(data_classes)):
    bal_acc = 0
    for j in range(n_splits):
        bal_acc += (acc0all[i][j] + acc1all[i][j])/(2*n_splits) # avg

    print("Avg balanced accuracy for " + data_classes[i] + ": ", bal_acc)
    print('\tClass 0: {}'.format(np.mean(acc0all[i])))
    print('\tClass 1: {}'.format(np.mean(acc1all[i])))
    print("Avg loss: ", np.mean([item[0] for item in evalall[i]]),
    ", avg acc: ", np.mean([item[1] for item in evalall[i]]))

# Now for the actual bag statistics
acc0all_bag = [[None]*n_splits for i in range(len(data_classes))]
acc1all_bag = [[None]*n_splits for i in range(len(data_classes))]
threads = [threading.Thread(target=test_bag_level, args=(i,
                                                         n_splits,
                                                         copy.deepcopy(alldata[i // n_splits]),
                                                         history[i // n_splits][i % n_splits],
                                                         0.5))
           for i in range(len(data_classes)*n_splits)]  # do this for all iterations of the model I suppose

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()

print("After {} epochs:".format(epochs))
for i in range(len(data_classes)):
    bal_acc = 0
    for j in range(n_splits):
        bal_acc += (acc0all_bag[i][j] + acc1all_bag[i][j])/(2*n_splits)  # avg

    print("Avg balanced bag accuracy for class " + str(i) + " (" + data_classes[i] + "): ", bal_acc)
    print('\tClass 0: {}'.format(np.mean(acc0all_bag[i])))
    print('\tClass 1: {}'.format(np.mean(acc1all_bag[i])))

"""
for i, data_class in enumerate(data_classes):
    data_class_path = os.path.join(SIVAL_data_dir, data_class)
    alldata[i][0] = data_class.split(".")[0]
    alldata[i][1] = getrawdata(data_class_path)
    print("Read data from " + alldata[i][0])
"""


"""
alldata is now a horrid contraption where:
    alldata[i][0] is the filename/easy to read descriptor
    alldata[i][1][0] is the bag id
    alldata[i][1][1] contains the data, first column being instance id, last one instance label, feature vectors between

"""

"""
About the SIVAL data: each .data file contains several bags (images) with multiple instances (image segments) plus
their feature vectors and a final feature signifying whether or not the instance belongs to the positive class (implying
that all bags in these datasets are positive for their own label - see also README.TXT).

Settles et al. train their model with positive bags from the selected category, as well as negative bags
(= bags from other image sets).

SO, step by step:
Research question is whether knowing bag labels can lead to improved instance labels.
As such:
 - Train a model to predict instances only. Evaluate performance at instance level and at bag level
    - Dataset: both positive instances from the image category (.data file) and random instances from this and other
               categories
 - Train a model to directly predict bag labels. Preferably score it against the previous, it should do better
    - Dataset: both bags from the image category and random bags from other categories
    - Discard individual instance labels/replace them all by the bag label.
 - Using this model:
    - Append the predicted bag label to the instance feature vector, then
    - Train a model to predict instances only. See what gives when using the original model vs letting it optimise again
    - Evaluate performance at instance level (and at bag level but less relevant?)

Model in this case is a NN with residual connections
"""

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
