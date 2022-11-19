import numpy as np
import pickle
import tensorflow as tf
import threading

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

threadno = 16

# globals for output
evalall = []
history = []

lock = threading.Lock()  # for file i/o, though it probably makes sense to apply a mutex to the arrays being written too


def getdata(musk_csv, random_state):
    # data
    global X
    global y

    """ # Titanic
    X = np.genfromtxt(titanic, delimiter=",")
    y = np.genfromtxt(titanic_labels, delimiter=",")
    """

    # Musk 1
    # first read data as strings so we can split into a 3 level array by molecule name (would pandas be handier for this?)
    X_raw = np.genfromtxt(musk_csv, delimiter=";", skip_header=1, dtype=str)

    y_grouped, s = np.unique(X_raw[:, 1], return_index=True)  # bag names, index where we go to a new bag
    X_grouped = np.split(X_raw,
                         s[1:])  # returns a list of 102 arrays of dimension (n, 166) where n is the number of confs

    # convert to np array and add column with bag labels, needed for sss
    X_grouped = np.asarray(X_grouped, dtype=object)
    X_grouped = np.c_[X_grouped, np.zeros(len(X_grouped))]

    for i, bag in enumerate(X_grouped):
        # if first instance of bag is labeled 1 (in the dataset all members of a positive bag are labeled 1)
        if int(bag[0][0][-1]) == 1:
            X_grouped[i, 1] = 1

    # We need to choose random items from the _list_ above for our training/test split, i.e. stratify bags, not instances

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=random_state)
    splits = sss.split(X_grouped[:, 0], X_grouped[:, 1])

    # must be a better way to access only one (multiprocessing thread id randomises these but don't need to return 10)
    X_train, X_test = [], []
    for train_index, test_index in splits:
        X_train, X_test = X_grouped[train_index, 0], X_grouped[test_index, 0]

    # flatten to 2D array from list of arrays, get labels, extract molecule name, strip non-feature columns, scale
    X_train, X_test = np.concatenate(X_train), np.concatenate(X_test)
    y_train, y_test = X_train[:, -1].astype(int), X_test[:, -1].astype(int)
    train_bag, test_bag = X_train[:, 1], X_test[:, 1]
    X_train, X_test = np.delete(X_train, (0, 1, 2, -1), 1).astype(int), np.delete(X_test, (0, 1, 2, -1), 1).astype(int)
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)

    return X_train, X_test, y_train, y_test, train_bag, test_bag


def mi_Net(nfeat):
    model = tf.keras.models.Sequential([  # as per Wang et al., but without dropout
        # to do: add kernel regularizers
        tf.keras.layers.Dense(units=256, activation='ReLU', input_dim=nfeat),
        tf.keras.layers.Dense(units=128, activation='ReLU'),
        tf.keras.layers.Dense(units=64, activation='ReLU'),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
        tf.keras.layers.MaxPooling1D(pool_size=2)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model


def fit_test(tid, model, x_train, x_test, y_train, y_test, train_bag, test_bag):
    # understand parameters a bit better here
    history.append(model.fit(x_train, y_train, epochs=10, batch_size=30, validation_split=.2))

    # instance level prediction. This needs to be flattened/pooled into bag level, will do this later
    threshold = .5
    classes = (model.predict(x_test, batch_size=20) > threshold).astype(int)
    # effectively the pooling layer... Though outside the model, so probably not the same approach?

    # assign bag labels based on instance label
    bag_pred = np.zeros(len(np.unique(test_bag)))
    bag_gt = np.zeros(len(np.unique(test_bag)))
    for i, bag_label in enumerate(np.unique(test_bag)):
        # get predictions per bag
        preds = classes[np.where(bag_label == test_bag)]
        gt = y_test[np.where(bag_label == test_bag)]

        bag_pred[i] = np.max(preds)
        bag_gt[i] = np.max(gt)

    # note that the following is in the instance level evaluation
    evals = model.evaluate(x_test, y_test, batch_size=20)
    # print('Loss: {}'.format(evals[0]))
    # print('Accuracy: {}'.format(evals[1]))
    evalall.append(evals)

    idx0 = (bag_gt == 0)
    idx1 = (bag_gt == 1)

    acc0 = np.mean(np.equal(bag_pred[idx0], bag_gt[idx0]))
    acc0all[tid] = acc0
    acc1 = np.mean(np.equal(bag_pred[idx1], bag_gt[idx1]))
    acc1all[tid] = acc1
    #bal_acc.append((acc0 + acc1) / 2)
    #print('Balanced accuracy: {}'.format(bal_acc))
    #print('\tClass 0: {}'.format(acc0all))
    #print('\tClass 1: {}'.format(acc1all))

    return history, acc0all, acc1all


def threadwrapper(tid):
    X_train, X_test, y_train, y_test, train_bag, test_bag = getdata("../input/musk_csv.csv", random_state=tid)
    nfeat = X_train.shape[1]
    model = mi_Net(nfeat)

    # containers for accuracies
    global acc0all
    global acc1all
    acc0all = np.zeros(threadno)
    acc1all = np.zeros(threadno)
    fit_test(tid, model, X_train, X_test, y_train, y_test, train_bag, test_bag)


def musk_mi():
    threads = [threading.Thread(target=threadwrapper, args=(i, )) for i in range(threadno)]

    for i, thread in enumerate(threads):
        thread.start()

    for i, thread in enumerate(threads):
        thread.join()

    bal_acc = 0
    for i in range(0, len(acc0all)):
        bal_acc += (acc0all[i] + acc1all[i])/(2*threadno) # avg

    print("Avg balanced accuracy: ", bal_acc)
    print('\tClass 0: {}'.format(np.mean(acc0all)))
    print('\tClass 1: {}'.format(np.mean(acc1all)))
    print("Avg loss: ", np.mean([item[0] for item in evalall]),
    ", avg acc: ", np.mean([item[1] for item in evalall]))

    f = open("../output/vardump", mode='wb')
    pickle.dump(dir(), f)
    f.close()

    return dir()