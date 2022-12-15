import data_proc
import ml_models_inst
import ml_models_bag
import numpy as np
import os
import pickle
import tensorflow as tf

import models_weighted

"""
PURPOSE OF THIS FILE:
A simple sanity check whether appending the bag labels to the instance feature vectors can result in better
instance prediction.
This is equivalent to having a perfect bag prediction model (and then predicting the instances).
The same simple model is used for both, a 4 layer fc NN (256 > 128 > 64 > n_image_classes + 1.
We can add dropouts and regularisation later if need be.
"""

SIVAL_data_dir = "../input/amil-sival/"

# Don't redo reading in the data every time since it takes a while
if (os.path.exists("input_data.pickle")):
    pickle_in = open("input_data.pickle", "rb")
    data_dict = pickle.load(pickle_in)

    data_flat = data_dict["data_flat"]
    bag_labels = data_dict["bag_labels"]
    data_bagged = data_dict["data_bagged"]
else:
    data_flat = data_proc.get_data(SIVAL_data_dir, ".data")

    bag_labels, data_bagged = data_proc.split_into_bags(data_flat, bag_id_column=1)

    data_dict = {"data_flat": data_flat,
             "bag_labels": bag_labels,
             "data_bagged": data_bagged}

    pickle_out = open("input_data.pickle", "wb")
    pickle.dump(data_dict, pickle_out)

n_splits = 10
omit_first_cols = 3
epochs = 250

train_test_data = data_proc.split_and_proc_instances(n_splits,
                                                     data_flat,
                                                     omit_first_cols=omit_first_cols,
                                                     verbose=True)
n_feat = train_test_data[0].X_train.shape[1] - omit_first_cols
n_classes = np.max(data_flat[:, -1])

model = ml_models_inst.baseline_inst_model(n_feat, n_classes)

history_before, evals_before, acc_before = ml_models_inst.fit_and_test(n_splits=n_splits,
                                                                       model=model,
                                                                       data=train_test_data,
                                                                       epochs=epochs,
                                                                       verbose=True)

# Train a model to predict bag labels
bag_model = ml_models_bag.bag_model(n_feat, n_classes)
train_test_data_bagged = data_proc.train_test_split_bag(n_splits,
                                                        input_data=data_bagged,
                                                        data_labels=bag_labels,
                                                        validation_split=0.2,
                                                        omit_first_cols=omit_first_cols,
                                                        verbose=True)

history_bag, evals_bag, acc_bag = ml_models_bag.fit_and_test_bag(n_splits=n_splits,
                                                                 model=bag_model,
                                                                 data=train_test_data_bagged,
                                                                 epochs=epochs,
                                                                 verbose=True)
# Don't forget to preprocess. So flatten, normalise, resplit
data_bagged_prepr = data_proc.preprocess_bagged_data(data_bagged)

# CAVEAT: this approach probably gives us "perfect" results because of overfitting (i.e. each bag will be part of the
# training set for most models)
data_bagged_truncated =\
    tf.ragged.constant(np.asarray([bag[:, 3:-1] for bag in data_bagged_prepr], dtype='object'), ragged_rank=1)
bag_predictions = [history_bag[i].model.predict(
                        data_bagged_truncated)
                   for i in range(n_splits)]
bag_pred_array = [np.argmax(bag_predictions[i], axis=1) + 1 for i in range(n_splits)]
bag_pred_array = np.transpose(np.asarray(bag_pred_array))

bins = [np.bincount(img) for img in bag_pred_array]
predicted_bag = np.asarray([np.argmax(bin[1:]) + 1 for bin in bins])

# Now append the bag label to the instance feature vectors
data_bagged_labeled = [np.vstack(
        [
            np.hstack([instance[:-1], predicted_bag[i], instance[-1]])
        for instance in bag]
    )
    for i, bag in enumerate(data_bagged)]

# Flatten for simple instance classification
data_labeled_flat = np.vstack(data_bagged_labeled)

train_test_data_after = data_proc.split_and_proc_instances(n_splits,
                                                           data_labeled_flat,
                                                           omit_first_cols=omit_first_cols,
                                                           verbose=True)
n_feat = train_test_data_after[0].X_train.shape[1] - omit_first_cols
n_classes = np.max(data_labeled_flat[:, -1])

model_labeled = models_weighted.baseline_inst_model(n_feat, n_classes)

history_after, evals_after, acc_after = models_weighted.fit_and_test(n_splits=n_splits,
                                                                    model=model_labeled,
                                                                    data=train_test_data_after,
                                                                    epochs=epochs,
                                                                    verbose=True)

print("Avg balanced accuracy before adding bag labels: {:.4f}".format(np.mean(acc_before),
                                                        ", after: {:.4f}".format(np.mean(acc_after))))
print('\tClass 0: {:.4f}'.format(np.mean(acc_before[:, 0])) + ' +/- {:.4f}'.format(np.std(acc_before[:, 0])) +
      ', after: {:.4f}'.format(np.mean(acc_after[:, 0])) + ' +/- {:.4f}'.format(np.std(acc_after[:, 0])))
print('\tClass 1: {:.4f}'.format(np.mean(acc_before[:, 1])) + ' +/- {:.4f}'.format(np.std(acc_before[:, 1])) +
      ', after: {:.4f}'.format(np.mean(acc_after[:, 1])) + ' +/- {:.4f}'.format(np.std(acc_after[:, 1])))
print("Avg loss before: ", np.mean(evals_before[:, 0]), " after: ", np.mean(evals_after[:, 0]),
      ",\n avg unbalanced acc before: ", np.mean(evals_before[:, 1]), " after: ", np.mean(evals_after[:, 1]))

""" First attempt, baseline, w/ perfect prediction:
Avg balanced accuracy before adding bag labels: 0.8075 , after: 0.8919
	Class 0: 0.9733 +/- 0.0056, after: 0.9655 +/- 0.0058
	Class 1: 0.6418 +/- 0.0297, after: 0.8182 +/- 0.0231
Avg loss before:  0.42201949656009674  after:  0.23361250907182693 ,
 avg unbalanced acc before:  0.8885418832302093  after:  0.9294576942920685
 """

""" After adding layer weights, w/ perfect prediction:
Avg balanced accuracy before adding bag labels: 0.8286
	Class 0: 0.9592 +/- 0.0060, after: 0.9586 +/- 0.0058
	Class 1: 0.6980 +/- 0.0130, after: 0.8436 +/- 0.0163
Avg loss before:  0.47405731678009033  after:  0.26115474849939346 ,
 avg unbalanced acc before:  0.8925511717796326  after:  0.9310192048549653
 """
