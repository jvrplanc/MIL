import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

def inst_model(n_feat: int, n_classes: int):
    """
    Takes a numpy array of n_feat columns and outputs a vector of length n_classes + 1 with probabilities for each class.

    :param n_feat: Number of features in the input array.
    :param n_classes: Number of positive classes. Total number of classes output is n_classes + 1 for the extra negative
    class.
    :return:

    Prediction of instance labels without bag label is quite poor (~40% precision), in contrast to when we run the model
    on the entire dataset (~75%), indicative of some limitations of the synthetically generated data.
    """
    model = tf.keras.models.Sequential([
        Dense(units=256, activation='ReLU', name='fc_256', input_dim=n_feat, kernel_regularizer='l2'),
        #Dropout(rate=0.2, name='dropout'),
        Dense(units=128, name='fc_128', activation='ReLU'),
        Dense(units=64, name='fc_64', activation='ReLU'),
        Dense(units=n_classes+1, name='label_predictions', activation='sigmoid')
    ])

    return model


def instpred_instloss(input_bag, model, omit_first_cols):
    input_bag_trunc = input_bag[:, omit_first_cols:-1].astype('float32')
    inst_predictions = model.predict(input_bag_trunc)

    if np.all(np.argmax(inst_predictions, axis=1)) == 0:
        bag_label = 0
    else:
        bag_label = np.argmax(np.bincount(np.argmax(inst_predictions, axis=1))[1:]) + 1

    return bag_label


def minn_instpred_bagloss(n_feat, n_classes):
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

    inputs = Input(shape=(None, n_feat), name="input")  # variable number of instances per bag

    # Build instance model
    instance_model = inst_model(n_feat, n_classes)  # None, n_classes + 1

    instance_output = TimeDistributed(instance_model, name="distribution_layer")(inputs)  # None, None, n_classes + 1

    # Condense into bag prediction
    bag_output = GlobalAveragePooling1D(name="pooling")(instance_output)  # Max of all positive instances = bag label
    bag_output = Lambda(lambda x: x[:, 1:], name="label_selection")(bag_output)  # Drop item 0 because not possible for a bag... Or should I?

    model = tf.keras.models.Model(inputs, bag_output)  # output is vector with predictions
    model.trainable = True

    return model


def minn_rc_bagpred_bagloss(n_feat, n_classes):

    inputs = Input(shape=(None, n_feat))  # variable number of instances per bag

    dense1 = Dense(units=256, activation='ReLU', input_dim=n_feat, name="fc_256")
    # https://github.com/tensorflow/tensorflow/issues/39072
    dense1_wrapper = TimeDistributed(dense1, name="fc_256_wrapper")(inputs)  # allows passing ragged tensor into dense layer
    dense2 = Dense(units=128, activation='ReLU', name="fc_128")
    dense3 = Dense(units=64, activation='ReLU', name="fc_64")

    pool1 = GlobalMaxPooling1D(name="pool3")
    pool2 = GlobalMaxPooling1D(name="pool2")
    pool3 = GlobalMaxPooling1D(name="pool1")

    pred1 = Dense(units=n_classes + 1, activation='softmax', name="bag_feat3")
    pred2 = Dense(units=n_classes + 1, activation='softmax', name="bag_feat2")
    pred3 = Dense(units=n_classes + 1, activation='softmax', name="bag_feat1")

    path1 = pred1(pool1(dense3(dense2(dense1_wrapper))))
    path2 = pred2(pool2(dense2(dense1_wrapper)))
    path3 = pred3(pool3(dense1_wrapper))

    concat = Concatenate(axis=-1, name="aggregation_layer")([path1, path2, path3])
    bag_pred = Dense(units=n_classes+1, name="weighting_layer", activation='softmax')(concat)  # weighted output
    bag_label = Lambda(lambda x: x[:, 1:], name="label_selection")(bag_pred)
    model = tf.keras.models.Model(inputs, bag_label)

    model.trainable = True

    return model


def minn_ds_bagpred_bagloss(n_feat, n_classes):
    inputs = Input(shape=(None, n_feat))  # variable number of instances per bag

    dense1 = Dense(units=256, activation='ReLU', input_dim=n_feat, name="fc_256")
    # https://github.com/tensorflow/tensorflow/issues/39072
    dense1_wrapper = TimeDistributed(dense1, name="fc_256_wrapper")(inputs)  # allows passing ragged tensor into dense layer
    dense2 = Dense(units=128, activation='ReLU', name="fc_128")
    dense3 = Dense(units=64, activation='ReLU', name="fc_64")

    pool1 = GlobalMaxPooling1D(name="pool3")
    pool2 = GlobalMaxPooling1D(name="pool2")
    pool3 = GlobalMaxPooling1D(name="pool1")

    pred1 = Dense(units=n_classes + 1, activation='softmax', name="bag_feat3")
    pred2 = Dense(units=n_classes + 1, activation='softmax', name="bag_feat2")
    pred3 = Dense(units=n_classes + 1, activation='softmax', name="bag_feat1")

    path1 = pred1(pool1(dense3(dense2(dense1_wrapper))))
    path2 = pred2(pool2(dense2(dense1_wrapper)))
    path3 = pred3(pool3(dense1_wrapper))

    bag_pred = Average(name="bag_pooler")([path1, path2, path3])  # average of all predictions from each path
    bag_label = Lambda(lambda x: x[:, 1:], name="label_selection")(bag_pred)
    model = tf.keras.models.Model(inputs, bag_label)

    model.trainable = True

    return model


def cnn(n_feat, n_classes):
    CNN_model = tf.keras.Sequential([
        # don't forget to add the extra dimension to your input data
        tf.keras.layers.Input(shape=(32, n_feat, 1), ragged=False),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', kernel_regularizer='l2'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.25),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer='l2'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer='l2'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='ReLU'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    CNN_model.trainable = True

    return CNN_model
