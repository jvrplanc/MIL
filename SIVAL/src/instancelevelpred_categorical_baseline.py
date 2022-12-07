import copy
import numpy as np
import os
#import pickle
import shelve
import sys
import tensorflow as tf
import threading

from collections import namedtuple
from keras.utils.np_utils import to_categorical
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


def get_rawdata(inputfile, filter):
    # for SIVAL data, first column is the image name (bag name).
    # A better approach is probably to get this data for all classes, then merge it such that we have multi hot encoding
    names = np.genfromtxt(inputfile, delimiter=",", usecols=0, dtype=str)
    keep = np.char.find(np.char.lower(names), filter.lower()) > 0
    names = names[keep]
    data = np.genfromtxt(inputfile, delimiter=",", dtype=float)[:, 1:]
    data = data[keep]
    return names, data


def get_data_wrapper(thread_id, classes, data):
    # For categorical loss function:
    #     - We don't need the entire dataset but only those belonging to the observed positive class
    #       (We'll concatenate later)
    #     - Update the instance labels to an integer on the range (0, len(classes)) for later categorical labeling
    data_class = classes[thread_id]
    data_class_path = os.path.join(SIVAL_data_dir, data_class)
    name = data_class.split(".")[0]
    data[thread_id][0] = name
    data[thread_id][1] = get_rawdata(data_class_path, name)
    # We now update the instance labels to an integer on the range (0, len(classes)) for later categorical labeling
    data[thread_id][1][1][np.where(data[thread_id][1][1][:, -1] == 1), -1] = thread_id + 1
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


def nn_model_inst(nfeat, n_classes):
    model = tf.keras.models.Sequential([  # as per Wang et al., but without dropout
        # to do: add kernel regularizers
        tf.keras.layers.Dense(units=256, activation='ReLU', input_dim=nfeat),
        tf.keras.layers.Dense(units=128, activation='ReLU'),
        tf.keras.layers.Dense(units=64, activation='ReLU'),
        tf.keras.layers.Dense(units=n_classes+1, activation='sigmoid')  # ,
        # tf.keras.layers.MaxPooling1D(pool_size=2)
    ])

    # comment this out?
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def process_data_wrapper(thread_id, no_splits, data, names):
    print("Starting thread to generate dataset for class " + str(thread_id))

    # From data, extract X and y. First col is the instance id, last the instance label, rest contains feature vectors
    # Really need to be more consistent with array shapes and sizes, the one times it's a 1D array of n_splits*len(data),
    # the next it's a 2D array with shape n_splits, len(data). Error-prone!
    class_id = thread_id
    X = data[:, 1:-1]
    y = data[:, -1]

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
                                         args=(i, thread_info, data, names, indices))

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
    mi_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])  # Categorical probably makes the most sense

    history[split_no] = \
        mi_model.fit(data.X_train, to_categorical(data.y_train), epochs=no_epochs, validation_split=0.2, batch_size=batch_size)

    predictions = mi_model.predict(data.X_test, batch_size=batch_size)
    """for prediction in predictions:
        argmax = np.argmax(prediction)
        prediction[:] = 0
        prediction[argmax] = 1"""

    evals = mi_model.evaluate(data.X_test, to_categorical(data.y_test), batch_size=batch_size)
    # print('Loss: {}'.format(evals[0]))
    # print('Accuracy: {}'.format(evals[1]))
    evalall[split_no] = evals

    maxpos = lambda x : np.argmax(x)

    y_true_max = np.array([maxpos(row) for row in to_categorical(data.y_test)])
    y_pred_max = np.array([maxpos(row) for row in predictions])

    CalculatedCategoricalAccuracy = sum(y_pred_max == y_true_max)/len(y_true_max)

    # print("Calculated Categorical Accuracy: " + str(CalculatedCategoricalAccuracy))
    metric = tf.keras.metrics.CategoricalAccuracy()
    metric.update_state(to_categorical(data.y_test), predictions)
    # print("Categorical accuracy: " + str(metric.result().numpy()))

    idx0 = np.where(y_true_max == 0)
    idx1 = np.where(y_true_max != 0)

    y0_true = y_true_max[idx0]
    y1_true = y_true_max[idx1]

    y0_pred = y_pred_max[idx0]
    y1_pred = y_pred_max[idx1]

    acc0 = np.mean(np.equal(y0_true, y0_pred))
    acc1 = np.mean(np.equal(y1_true, y1_pred))
    acc0all[thread_id] = acc0
    acc1all[thread_id] = acc1

    bal_acc = np.mean([acc0, acc1])
    # print("Balanced accuracy: " + str(bal_acc))

    return


def test_bag_level(thread_id, no_splits, data, model_history):
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
    unique, s = np.unique(data[:, 1], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s)
    unique, s = unique[indices], s[indices]  # sorted list of indices indicating start of new image.

    data_grouped = np.split(data, s[1:])
    bag_data = [np.concatenate((data_grouped[i][0][:-1], [np.max(data_grouped[i][:, -1])])) for i in range(len(unique))]
    bag_data = np.asarray(bag_data, dtype='object')
    if np.equal(bag_data[:, 1], unique).all() == False:
        pass
    bag_gt = np.delete(bag_data, np.s_[2:-1], axis=1)

    # Run the whole dataset through the model. Discard instance id and labels of course...
    instance_data = data[:, 3:-1]
    # Does preprocessing lead to data leakage in this case,or some kind of invalid test?
    instance_data = preprocessing.scale(instance_data)

    local_model = model_history.model
    inst_predictions = local_model.predict(instance_data) # is this in the same order as bag_data?
    inst_predictions = np.asarray([np.argmax(pred) for pred in inst_predictions])
    inst_predictions_grouped = np.split(inst_predictions, s[1:])
    # bag label = most prevalent non-zero number
    # np.bincount returns the bin size of all integers starting from zero, for a given image i
    # drop the first element to figure out which instance label is the most prevalent
    # add 1 to compensate for this
    global bag_predictions
    bag_predictions = np.empty((len(unique), 2)).astype('object')
    for i, image_name in enumerate(unique):
        bins = np.bincount(inst_predictions_grouped[i])
        bag_predictions[i, 0] = image_name
        if len(bins) > 1:
            bag_predictions[i, 1] = np.argmax(bins[1:]) + 1
        else:
            bag_predictions[i, 1] = 0

    if np.equal(bag_predictions[:, 0], unique).all() == False:
        pass
    # per-class accuracy
    unique_classes, s = np.unique(bag_data[:, 0], return_index=True)  # returns positions of unique image classes
    indices = np.argsort(s)
    unique_classes, s = unique_classes[indices], s[indices]  # unsort the results of np.unique so we have the right order

    bag_predictions_split, bag_gt_split = np.split(bag_predictions[:, -1], s[1:]), np.split(bag_gt[:, -1], s[1:])

    for i, img_class_pred in enumerate(bag_predictions_split):
        idx0 = (img_class_pred == 0)
        idx1 = (img_class_pred != 0)
        img_class_gt = bag_gt_split[i].astype('float').astype('int')

        acc0 = np.mean(np.equal(img_class_pred[idx0], img_class_gt[idx0]))
        acc0all_bag[i][split_no] = acc0
        acc1 = np.mean(np.equal(img_class_pred[idx1], img_class_gt[idx1]))
        acc1all_bag[i][split_no] = acc1

        name_class[i][split_no] = unique_classes[i]

    return

def predict_bag_label(thread_id, data, model_history):
    unique_img, s = np.unique(data[0], return_index=True)
    indices = np.argsort(s)
    unique_img, s = unique_img[indices], s[indices]

    data_bagged, names_bagged = np.split(data[1], s[1:]), np.split(data[0], s[1:])

    local_model = model_history.model
    for i, bag in enumerate(data_bagged):
        prediction = local_model.predict(bag[:, 1:-1])
        pred_array = np.asarray([np.argmax(pred) for pred in prediction])
        bins = np.bincount(pred_array)
        predicted_bag = np.argmax(bins[1:]) + 1
        pred_array = np.full((len(bag), 1), predicted_bag)
        # temporary: write the correct bag label and see if that helps --> baseline, this helps:
        """
        Avg balanced accuracy after adding bag labels: 0.8901910855129181 , before: 0.8122013061375652
            Class 0: 0.9638116846284742, before: 0.9637266023823029
            Class 1: 0.816570486397362, before: 0.6606760098928277
        Avg loss:  0.2827687129378319 , avg acc:  0.9298375189304352
        """
        """
        actual_bag = thread_id + 1
        pred_array = np.full((len(bag), 1), actual_bag)
        """
        # So this stands or falls with having accurate bag predictions!

        bag = np.hstack((bag[:, :-1], pred_array, bag[:, -1:]))
        data_bagged[i] = bag

    #alldata[:][1] became a tuple somehow so need to do some magic
    alldata[thread_id][1] = list(alldata[thread_id][1])
    alldata[thread_id][1][1] = np.asarray(np.vstack(data_bagged), dtype='object')

    return


SIVAL_data_dir = "../input/amil-sival/"
data_classes = get_classes(SIVAL_data_dir, ".data")

# new 2 column np array, col 0 is data classes and col 1 the data. Not sure whether this is the most efficient way
alldata = [[0 for i in range(2)] for j in range(len(data_classes))]

# 25 threads for 25 image classes, get these in parallel
threads = [threading.Thread(target=get_data_wrapper, args=(i, data_classes, alldata)) for i in range(len(data_classes))]

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()

# concatenate data (yeah, the way I'm manipulating data is overly convoluted...)
alldata_concat = [alldata[i][1][1] for i in range(len(alldata))]
allnames_concat = [alldata[i][1][0] for i in range(len(alldata))]
alldata_concat = np.vstack(alldata_concat)
allnames_concat = np.concatenate(allnames_concat)

# Categorical labels
# WARNING: in this scheme, negative instances correspond to [1 0 0 0 ...] !
""" # Do this later, for comparison
cat_labels = to_categorical(alldata_concat[:, -1]).astype(int)
for i in range(len(alldata_concat)):
    alldata_concat[i, -1] = cat_labels[i]
"""

# generate 10 datasets
n_splits = 10

train_test_data = [None] * n_splits
process_data_wrapper (0, n_splits, alldata_concat, allnames_concat) # just the one, multithr is useless

# Not sure whether it makes sense to start and restart so many parallel threads vs just keeping it all in 1 thread
# per test/training split...
# We now have n_splits*no_classes datasets (250), all preprocessing.scaled.
model = nn_model_inst(nfeat=train_test_data[0][0].shape[1], n_classes=len(data_classes))
epochs = 25

history = [None]*n_splits
evalall = [None]*n_splits
acc0all = [None]*n_splits
acc1all = [None]*n_splits

threads = [threading.Thread(target=fit_and_test, args=(i, n_splits, tf.keras.models.clone_model(model),
                                                       train_test_data[i], epochs))
           for i in range(len(train_test_data))]

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()


bal_acc = 0
for j in range(n_splits):
    bal_acc += (acc0all[j] + acc1all[j])/(2*n_splits) # avg

print("Avg balanced accuracy: ", bal_acc)
print('\tClass 0: {}'.format(np.mean(acc0all)))
print('\tClass 1: {}'.format(np.mean(acc1all)))
print("Avg loss: ", np.mean([item[0] for item in evalall]),
", avg acc: ", np.mean([item[1] for item in evalall]))

acc0all_before = acc0all
acc1all_before = acc1all

# Now for the actual bag statistics
"""
acc0all_bag = [[None]*n_splits for i in range(len(data_classes))]
acc1all_bag = [[None]*n_splits for i in range(len(data_classes))]
name_class = [[None]*n_splits for i in range(len(data_classes))]

names = np.empty(shape=(len(alldata_concat), 1)).astype('str')
prevstart = 0
for i in range(len(data_classes)):
    names[prevstart:prevstart+len(alldata[i][1][0])] = alldata[i][0]
    prevstart += len(alldata[i][1][0])

# No awards for most elegant code here. Appends the image names to the data in the first column
temp = [None]*(np.shape(alldata[0][1][1])[1] + 2)
for i in range(len(data_classes)):
    joined_data = np.c_[[alldata[i][0] for item in alldata[i][1][0]], alldata[i][1][0]]
    joined_data = np.c_[joined_data, alldata[i][1][1]]
    temp = np.vstack((joined_data, temp))

data_bags = temp[:-1]

# Run the data through the different models
bag_predictions = []
threads = [threading.Thread(target=test_bag_level, args=(i,
                                                         n_splits,
                                                         data_bags,
                                                         history[i]
                                                         ))
           for i in range(n_splits)]  # do this for all iterations of the model I suppose

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()

print("After {} epochs (NaN = irrelevant):".format(epochs))
for i in range(len(data_classes)):
    bal_acc = 0
    for j in range(n_splits):
        bal_acc += (acc0all_bag[i][j] + acc1all_bag[i][j])/(2*n_splits)  # avg

    print("Avg balanced bag accuracy for class " + str(i) + " (" + str(name_class[i][j]) + "): ", bal_acc)
    print('\tClass 0: {}'.format(np.mean(acc0all_bag[i])))
    print('\tClass 1: {}'.format(np.mean(acc1all_bag[i])))
"""
# now we predict all bag labels and append them to each instance.
threads = [threading.Thread(target=predict_bag_label, args=(i, alldata[i][1], history[0]))
           for i in range(len(data_classes))]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

# now train a new model with this extra data
# concatenate data (yeah, the way I'm manipulating data is overly convoluted...)
alldata_concat = [alldata[i][1][1] for i in range(len(alldata))]
allnames_concat = [alldata[i][1][0] for i in range(len(alldata))]
alldata_concat = np.vstack(alldata_concat)
allnames_concat = np.concatenate(allnames_concat)

# generate 10 datasets
n_splits = 10

train_test_data = [None] * n_splits
process_data_wrapper (0, n_splits, alldata_concat, allnames_concat) # just the one, multithr is useless

model = nn_model_inst(nfeat=train_test_data[0][0].shape[1], n_classes=len(data_classes))

history = [None]*n_splits
evalall = [None]*n_splits
acc0all = [None]*n_splits
acc1all = [None]*n_splits

threads = [threading.Thread(target=fit_and_test, args=(i, n_splits, tf.keras.models.clone_model(model),
                                                       train_test_data[i], epochs))
           for i in range(len(train_test_data))]

for i, thread in enumerate(threads):
    thread.start()
for i, thread in enumerate(threads):
    thread.join()


bal_acc = 0
bal_acc_before = 0
for j in range(n_splits):
    bal_acc += (acc0all[j] + acc1all[j])/(2*n_splits) # avg
    bal_acc_before += (acc0all_before[j] + acc1all_before[j])/(2*n_splits) # avg

print("Avg balanced accuracy after adding bag labels: {}".format(bal_acc), ", before: {}".format(bal_acc_before))
print('\tClass 0: {}'.format(np.mean(acc0all)) + ', before: {}'.format(np.mean(acc0all_before)))
print('\tClass 1: {}'.format(np.mean(acc1all)) + ', before: {}'.format(np.mean(acc1all_before)))
print("Avg loss: ", np.mean([item[0] for item in evalall]),
", avg acc: ", np.mean([item[1] for item in evalall]))

# to do: check that all labels of bag test are in the right order
# to do: create loss function based on bag labels
# to do: create array for instance ground truth, before and after predictions

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