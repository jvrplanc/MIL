import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
import threading

from tensorflow.keras.layers import *


def baseline_inst_model(n_feat: int, n_classes: int):
    """
    Takes a numpy array of n_feat columns and outputs a vector of length n_classes + 1 with probabilities for each class.

    :param n_feat: Number of features in the input array.
    :param n_classes: Number of positive classes. Total number of classes output is n_classes + 1 for the extra negative
    class.
    :return:
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='ReLU', input_dim=n_feat),
        tf.keras.layers.Dense(units=128, activation='ReLU'),
        tf.keras.layers.Dense(units=64, activation='ReLU'),
        tf.keras.layers.Dense(units=n_classes+1, activation='sigmoid')
    ])

    return model


def fit_test_baseline(thread_id, model, data, no_epochs, return_data, verbose=False):
    if verbose:
        verbose_flag = 1
    else:
        verbose_flag = 0

    batch_size = 20

    # Get appropriate parts of data, cast
    X_train = data.X_train[:, 3:].astype('float32')
    X_test = data.X_test[:, 3:].astype('float32')

    y_train = data.y_train.astype('float32')
    y_test = data.y_test.astype('float32')

    sample_counts = np.bincount(y_train.reshape(len(y_train)).astype('int64'))
    sample_weights = np.array([1/np.log(sample_counts[label]) for label in y_train.astype('int32')])

    # recompile model as they're all clones
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  weighted_metrics=[tf.keras.metrics.CategoricalAccuracy()])  # Categorical probably makes the most sense

    es = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                          mode='auto',
                                          verbose=verbose_flag,
                                          patience=10,
                                          restore_best_weights=True)

    return_data["history"][thread_id] = \
        model.fit(X_train, to_categorical(y_train),
                  epochs=no_epochs,
                  validation_split=0.2,
                  batch_size=batch_size,
                  callbacks=[es],
                  sample_weight=sample_weights,
                  verbose=verbose_flag)

    return_data["evals"][thread_id] = model.evaluate(X_test,
                                                     to_categorical(y_test),
                                                     batch_size=batch_size,
                                                     verbose=verbose_flag)

    predictions = model.predict(X_test, batch_size=batch_size)

    maxpos = lambda x: np.argmax(x)

    y_true_max = y_test.reshape((len(y_test)))
    y_pred_max = np.array([maxpos(row) for row in predictions])

    # CalculatedCategoricalAccuracy = sum(y_pred_max == y_true_max)/len(y_true_max)
    # print("Calculated Categorical Accuracy: " + str(CalculatedCategoricalAccuracy))

    metric = tf.keras.metrics.CategoricalAccuracy()
    metric.update_state(to_categorical(y_test), predictions)

    if verbose:
        print("Categorical accuracy: " + str(metric.result().numpy()))

    idx0 = np.where(y_true_max == 0)
    idx1 = np.where(y_true_max != 0)

    y0_true = y_true_max[idx0]
    y1_true = y_true_max[idx1]

    y0_pred = y_pred_max[idx0]
    y1_pred = y_pred_max[idx1]

    acc0 = np.mean(np.equal(y0_true, y0_pred))
    acc1 = np.mean(np.equal(y1_true, y1_pred))

    return_data["accuracies"][thread_id] = [acc0, acc1]

    # bal_acc = np.mean([acc0, acc1])
    # print("Balanced accuracy: " + str(bal_acc))

    return


def fit_and_test(n_splits, model, data, epochs, verbose=False):
    # return data:
    history = [None] * n_splits
    evals = [None] * n_splits
    accuracies = [[None] * 2 for i in range(n_splits)]

    return_dict = { "history": history,
                    "evals": evals,
                    "accuracies": accuracies}

    threads = [threading.Thread(target=fit_test_baseline, args=(i, tf.keras.models.clone_model(model),
                                                                data[i], epochs, return_dict, verbose))
               for i in range(len(data))]

    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()

    # cast to np array for future ease, where appropriate
    return_dict["evals"] = np.asarray(return_dict["evals"])
    return_dict["accuracies"] = np.asarray(return_dict["accuracies"])

    return return_dict["history"], return_dict["evals"], return_dict["accuracies"]