import threading

import numpy as np
import tensorflow as tf

import data_proc
import utils


def fit_test_bag(thread_id, model, data, no_epochs, return_data, omit_first_cols=3, verbose=False, expand_dims=False):
    if verbose:
        verbose_flag = 1
    else:
        verbose_flag = 0

    batch_size = 20  # data.X_train//10 # this tripped up model.fit() for some as yet unknown reason

    # convert data to tensors first
    if expand_dims:
        X_train = utils.tensorify_and_expand(data.X_train, omit_first_cols)
        X_valid = utils.tensorify_and_expand(data.X_valid, omit_first_cols)
        X_test = utils.tensorify_and_expand(data.X_test, omit_first_cols)
    else:
        X_train = utils.tensorify(data.X_train, omit_first_cols)
        X_valid = utils.tensorify(data.X_valid, omit_first_cols)
        X_test = utils.tensorify(data.X_test, omit_first_cols)

    y_train = tf.convert_to_tensor(np.asarray(np.argmax(data.y_train, axis=1) - 1).astype('float32'))
    y_valid = tf.convert_to_tensor(np.asarray(np.argmax(data.y_valid, axis=1) - 1).astype('float32'))
    y_test = tf.convert_to_tensor(np.asarray(np.argmax(data.y_test, axis=1) - 1).astype('float32'))
    # Minus one because we drop the 0 (negative instance) category in the model

    # (re)compile model? as they're all clones
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # If sparse -> compare with single int in y
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])  # Categorical probably makes the most sense

    model.trainable = True

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='auto',
                                          verbose=verbose_flag,
                                          patience=20,
                                          restore_best_weights=True)
    return_data["history"][thread_id] = \
        model.fit(X_train, y_train,
                  epochs=no_epochs,
                  batch_size=batch_size,
                  validation_data=(X_valid, y_valid),
                  callbacks=[es],
                  verbose=verbose_flag)

    predictions = model.predict(X_test, batch_size=batch_size)

    evals = model.evaluate(X_test, y_test, batch_size=batch_size)
    # print('Loss: {}'.format(evals[0]))
    # print('Accuracy: {}'.format(evals[1]))
    return_data["evals"][thread_id] = evals

    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    metric.update_state(y_test, predictions)
    # print("Categorical bag accuracy: " + str(metric.result().numpy()))

    return


def fit_and_test_bag(n_splits, model, data, epochs, omit_first_cols=3, verbose=False, expand_dims=False):
    # return data:
    history = [None] * n_splits
    evals = [None] * n_splits
    accuracies = [[None] * 2 for i in range(n_splits)]

    return_dict = {"history": history,
                   "evals": evals,
                   "accuracies": accuracies}

    threads = [threading.Thread(target=fit_test_bag, args=(i, tf.keras.models.clone_model(model),
                                                           data[i], epochs, return_dict, omit_first_cols,
                                                           verbose, expand_dims))
               for i in range(len(data))]

    for i, thread in enumerate(threads):
        thread.start()
    for i, thread in enumerate(threads):
        thread.join()

    # cast to np array for future ease, where appropriate
    return_dict["evals"] = np.asarray(return_dict["evals"])
    return_dict["accuracies"] = np.asarray(return_dict["accuracies"])

    return return_dict["history"], return_dict["evals"], return_dict["accuracies"]
