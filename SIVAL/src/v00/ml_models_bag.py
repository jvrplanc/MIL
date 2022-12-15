import ml_models_inst
import numpy as np
import tensorflow as tf
import threading

from tensorflow.keras.layers import *


def bag_model(n_feat, n_classes):
    """
    This model takes (batch_size, n_instances, n_feat) as input and outputs a one-hot vector of length n_classes.
    Each instance in a bag is run through a separate NN (so the model optimises on individual instances and not bags).
    The NN for each instance is shared and thus has identical weights and biases - i.e. there is only one instance model.
    After assigning instance probabilities, the resulting vector is condensed into a single value (the predicted class).
    The predictions for all instances in a bag are pooled, taking the most prevalent non-zero prediction as the bag
    label. This bag label is then converted back into a one-hot vector representing the bag class.

    :param n_feat: Number of features.
    :param n_classes: Number of possible image classes (not including negative labels).
    :param n_instances: Number of instances per bag.
    :return: Compiled tf.keras.models.Model. Loss function, optimiser and metrics are fixed for now.
    """

    inputs = Input(shape=(None, n_feat))  # variable number of instances per bag

    # Build instance model
    inst_model = ml_models_inst.baseline_inst_model(n_feat, n_classes)  # None, n_classes + 1

    instance_output = TimeDistributed(inst_model)(inputs)  # None, None, n_classes + 1

    # Condense into bag prediction
    bag_output = GlobalAveragePooling1D()(instance_output)  # Max of all positive instances = bag label
    bag_output = Lambda(lambda x: x[:, 1:])(bag_output)  # Drop item 0 because not possible for a bag... Or should I?

    model = tf.keras.models.Model(inputs, bag_output)  # output is vector with predictions
    model.trainable = True

    return model


def fit_test_bag(thread_id, model, data, no_epochs, return_data, verbose=False):
    if verbose:
        verbose_flag = 1
    else:
        verbose_flag = 0

    batch_size = 20  # data.X_train//10 # this tripped up model.fit() for some as yet unknown reason

    # convert data to tensors first
    X_train = tf.ragged.constant(np.asarray([bag[:, 3:-1] for bag in data.X_train], dtype='object'), ragged_rank=1)
    X_valid = tf.ragged.constant(np.asarray([bag[:, 3:-1] for bag in data.X_valid], dtype='object'), ragged_rank=1)
    X_test = tf.ragged.constant(np.asarray([bag[:, 3:-1] for bag in data.X_test], dtype='object'), ragged_rank=1)
    y_train = tf.convert_to_tensor(np.asarray(np.argmax(data.y_train, axis=1)-1).astype('float32'))
    y_valid = tf.convert_to_tensor(np.asarray(np.argmax(data.y_valid, axis=1)-1).astype('float32'))
    y_test = tf.convert_to_tensor(np.asarray(np.argmax(data.y_test, axis=1)-1).astype('float32'))
    # Minus one because we drop the 0 (negative instance) category in the model

    # (re)compile model? as they're all clones
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # If sparse -> compare with single int in y
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])  # Categorical probably makes the most sense

    model.trainable = True

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='auto',
                                          verbose=verbose_flag,
                                          patience=10,
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
    #print("Categorical bag accuracy: " + str(metric.result().numpy()))

    return



def fit_and_test_bag(n_splits, model, data, epochs, verbose=False):
    # return data:
    history = [None] * n_splits
    evals = [None] * n_splits
    accuracies = [[None] * 2 for i in range(n_splits)]

    return_dict = { "history": history,
                    "evals": evals,
                    "accuracies": accuracies}

    threads = [threading.Thread(target=fit_test_bag, args=(i, tf.keras.models.clone_model(model),
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