import copy
import os
import pickle

import numpy as np

import data_proc
import fit_model_bag
import fit_model_inst
import models
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU faster than GPU on my machine

"""
PURPOSE OF THIS FILE:
"""

SIVAL_data_dir = "../input/amil-sival/"
synthetic = False

# Don't redo reading in the data every time since it takes a while
if os.path.exists("input_data.pickle"):
    pickle_in = open("input_data.pickle", "rb")
    data_dict = pickle.load(pickle_in)

    data_flat = data_dict["data_flat"]
    bag_labels = data_dict["bag_labels"]
    data_bagged = data_dict["data_bagged"]
    pickle_in.close()
else:
    data_flat = data_proc.get_data(SIVAL_data_dir, ".data", synthetic=synthetic)
    bag_labels, data_bagged = utils.split_into_bags(data_flat, bag_id_column=1)

    data_dict = {"data_flat": data_flat,
                 "bag_labels": bag_labels,
                 "data_bagged": data_bagged}

    pickle_out = open("input_data.pickle", "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()

# Bag model to train on half the bags. Test data can be other half
# With the other bags, k-fold split the data
# Predict the labels of the training datasets, test datasets, validation datasets, ... creating two arrays,
# one with labels and one without
# Run instance models on both. Reduces variation introduced by re-stratifying and keeps a fair amount of data available

# We split the bags 50/50 for now to avoid data leakage
# Could also use the one as the test set for the other to increase amount of data available for the models

# Could be args if we want to make this file callable
n_splits = 10
omit_first_cols = 3
max_epochs = 250
# possible values: ["theor_baseline", "instpred_instloss", "instpred_bagloss", "rc", "ds", "cnn"]
model = "theor_baseline"
theor_acc = 1

# Go halfsies. Mostly for comparability but from first tests it doesn't seem to introduce a systematic difference
# between datasets (and if it did, it would impact results negatively). A better approach would be to add an extra flag
# to the data and stratify on that.
# What we _do_ notice is an impact on the absolute accuracy of instance prediction, perhaps the synthetic data isn't
# capturing the trends properly, perhaps because it is being synthesised separately for each class (i.e. before
# concatenating all the different classes' data). Future work
# The above is not relevant anymore since we don't use synthetic data
"""
data_bagged_bm = data_bagged[::2]
bag_labels_bm = bag_labels[::2]
data_bagged_im = data_bagged[1::2]
bag_labels_im = bag_labels[1::2]
"""
# First 30 images are real, then synthetic, then real, etc
synth_column = np.array([bag[0, 2] for bag in data_bagged])

if synthetic:
    # This splits the bagged data into two sets, stratified on both synthetic/non synthetic and bag label
    data_bagged_im, data_bagged_bm, bag_labels_im, bag_labels_bm = \
        utils.split_real_synth_bags(synth_column, data_bagged, bag_labels)
else:
    # This splits the bagged data into two sets, stratified on bag label
    data_bagged_im, data_bagged_bm, bag_labels_im, bag_labels_bm = \
        utils.split_dataset_bags(data_bagged, bag_labels)

# using this has a huge effect on the instance level accuracy...
# data_bagged_im = np.ma.masked_array(data_bagged, np.array([bag[0,2] for bag in data_bagged])).compressed()
# data_bagged_bm = np.ma.masked_array(data_bagged, 1-np.array([bag[0,2] for bag in data_bagged])).compressed()

##### TRAIN A BAG MODEL - START #####
# Train a model to predict bag labels
# Split data. We do this first to dynamically pass the data's dimensions to the model
train_test_data_bagged = data_proc.split_proc_bag_wrapper(n_splits,
                                                          input_data=data_bagged_bm,
                                                          data_labels=bag_labels_bm,
                                                          validation_split=0.2,
                                                          omit_first_cols=omit_first_cols,
                                                          verbose=True)

n_feat = train_test_data_bagged[0].X_train[0].shape[1] - (omit_first_cols + 1)
n_inst = np.max(len(bag) for bag in data_bagged_bm)
n_classes = np.max(data_flat[:, -1])

if model == "theor_baseline":
    bag_model = None
    expand_dims = False
    bagacc_max = theor_acc
elif model == "instpred_instloss":
    bag_model = None
    expand_dims = False
elif model == "instpred_bagloss":
    bag_model = models.minn_instpred_bagloss(n_feat, n_classes)
    expand_dims = False
elif model == "rc":
    bag_model = models.minn_rc_bagpred_bagloss(n_feat, n_classes)
    expand_dims = False
elif model == "ds":
    bag_model = models.minn_ds_bagpred_bagloss(n_feat, n_classes)
    expand_dims = False
elif model == "cnn":
    bag_model = models.cnn(n_feat, n_classes)
    expand_dims = True
else:
    print("No valid bag model selected.")
    bag_model = []
    expand_dims = False
    quit()

if model != "theor_baseline" and model != "instpred_instloss":
    history_bag, evals_bag, acc_bag = fit_model_bag.fit_and_test_bag(n_splits=n_splits,
                                                                     model=bag_model,
                                                                     data=train_test_data_bagged,
                                                                     epochs=max_epochs,
                                                                     omit_first_cols=omit_first_cols,
                                                                     verbose=True,
                                                                     expand_dims=expand_dims)


    bagacc = np.mean([np.max(hist.history["val_sparse_categorical_accuracy"]) for hist in history_bag])
    bagacc_max = np.max([np.max(hist.history["val_sparse_categorical_accuracy"]) for hist in history_bag])
    bagacc_min = np.min([np.max(hist.history["val_sparse_categorical_accuracy"]) for hist in history_bag])
    print("Bag model avg categorical accuracy:", bagacc, ", max:", bagacc_max, ", min:", bagacc_min, "model:", model)

##### TRAIN A BAG MODEL - END #####


##### TRAIN INSTANCE MODEL WITHOUT LABELS - START #####

# Split data, flatten first as instance-level prediction doesn't care about which bag it belongs to
if model == "instpred_instloss":
    train_test_data = data_proc.split_proc_inst_wrapper(n_splits,
                                                        np.vstack(data_bagged_bm),
                                                        omit_first_cols=omit_first_cols,
                                                        validation_split=0.2,
                                                        verbose=True)
else:
    train_test_data = data_proc.split_proc_inst_wrapper(n_splits,
                                                        np.vstack(data_bagged_im),
                                                        omit_first_cols=omit_first_cols,
                                                        validation_split=0.2,
                                                        verbose=True)

n_feat = train_test_data[0].X_train.shape[1] - omit_first_cols
n_classes = np.max(data_flat[:, -1])

inst_model = models.inst_model(n_feat, n_classes)

history_before, evals_before, acc_before = fit_model_inst.fit_and_test(n_splits=n_splits,
                                                                       model=inst_model,
                                                                       data=train_test_data,
                                                                       epochs=max_epochs,
                                                                       omit_first_cols=omit_first_cols,
                                                                       verbose=True)
print("Finished fitting without bag labels.\n Statistics:")
report_before = utils.gather_stats(history_before, train_test_data, n_splits, omit_first_cols)

precision_before, recall_before, f1_before = \
    utils.parse_report(report_before, n_classes, print_output=True, with_stdev=True)

##### TRAIN INSTANCE MODEL WITHOUT LABELS - END #####


##### PREDICT BAG LABELS AND UPDATA DATASET - START #####
# Instance prediction with bag labels
# Don't forget to preprocess. So flatten, normalise, resplit!
# First get predictions for the instance data set
data_bagged_im_preproc = data_proc.preprocess_bagged_data(data_bagged_im, omit_first_cols)

if model == "cnn":
    data_bagged_truncated = utils.tensorify_and_expand(data_bagged_im_preproc, omit_first_cols)
else:
    data_bagged_truncated = utils.tensorify(data_bagged_im_preproc, omit_first_cols)

# Choose best model
if model == "theor_baseline":
    predicted_bag = np.array([np.max(bag[:, -1]) for bag in data_bagged_im])

    # for evaluation: what happens if we do _not_ have perfect prediction
    if theor_acc != 1:
        for i, label in enumerate(predicted_bag):
            if np.random.rand() > theor_acc:
                predicted_bag[i] = np.random.randint(1, 26)

elif model == "instpred_instloss":
    model_no = np.argmax([np.max(history.history["val_categorical_accuracy"]) for history in history_before])
    model_perf = np.amax([history.history["val_categorical_accuracy"] for history in history_before])
    model_inst = history_before[model_no].model

    predicted_bag = \
        np.array([models.instpred_instloss(bag, model_inst, omit_first_cols) for bag in data_bagged_im_preproc])
    bag_gt = np.array([np.max(bag[:, -1]) for bag in data_bagged_im_preproc])
    bagacc_max = np.mean(np.equal(predicted_bag, bag_gt))
else:
    model_no = np.argmax([np.max(history.history["val_sparse_categorical_accuracy"]) for history in history_bag])
    model_perf = np.amax([history.history["val_sparse_categorical_accuracy"] for history in history_bag])
    bag_predictions = history_bag[model_no].model.predict(data_bagged_truncated)
    predicted_bag = np.argmax(bag_predictions, axis=1) + 1


# optional check that the input for the bag prediction was properly preprocessed...
# predicted_bag_gt = np.asarray([np.max(bag[:, -1]) for bag in data_bagged_im])
# predicted_bag_acc = np.sum(np.equal(predicted_bag, predicted_bag_gt)) / len(predicted_bag)
# print("Model:", history_bag[i].history["val_sparse_categorical_accuracy"][-1], ", pred:", predicted_bag_acc)

predicted_bag_labels = [bag[0, 1] for bag in data_bagged_im]

if model == "instpred_instloss":
    train_test_data_labeled = data_proc.split_proc_inst_wrapper(n_splits,
                                                        np.vstack(data_bagged_im),
                                                        omit_first_cols=omit_first_cols,
                                                        validation_split=0.2,
                                                        verbose=True)
else:
    train_test_data_labeled = copy.deepcopy(train_test_data)

# For each split in the data, add the bag label to the X_train and X_test datasets
for i, dataset in enumerate(train_test_data_labeled):
    labeled_X_train = data_proc.append_bag_labels(dataset.X_train, predicted_bag_labels, predicted_bag)
    train_test_data_labeled[i] = train_test_data_labeled[i]._replace(X_train=labeled_X_train)

    labeled_X_test = data_proc.append_bag_labels(dataset.X_test, predicted_bag_labels, predicted_bag)
    train_test_data_labeled[i] = train_test_data_labeled[i]._replace(X_test=labeled_X_test)

    labeled_X_valid = data_proc.append_bag_labels(dataset.X_valid, predicted_bag_labels, predicted_bag)
    train_test_data_labeled[i] = train_test_data_labeled[i]._replace(X_valid=labeled_X_valid)

##### PREDICT BAG LABELS AND UPDATE DATASET - END #####

##### TRAIN INSTANCE MODEL WITH LABELS - START #####
# And run the model again with the extra info
n_feat = train_test_data_labeled[0].X_train.shape[1] - omit_first_cols

model_labeled = models.inst_model(n_feat, n_classes)

history_after, evals_after, acc_after = fit_model_inst.fit_and_test(n_splits=n_splits,
                                                                    model=model_labeled,
                                                                    data=train_test_data_labeled,
                                                                    epochs=max_epochs,
                                                                    omit_first_cols=omit_first_cols,
                                                                    verbose=True)
print("Finished fitting with bag labels")
report_after = utils.gather_stats(history_after, train_test_data_labeled, n_splits, omit_first_cols)

##### TRAIN INSTANCE MODEL WITH LABELS - END #####

##### DATA OUT #####
"""
data_dict = {"data_flat": data_flat,
             "bag_labels": bag_labels,
             "data_bagged": data_bagged,
             "precision_before": precision_before,
             "precision_after": precision_after,
             "recall_before": recall_before,
             "recall_after": recall_after,
             "bagacc": bagacc,
             "history_before": history_before,
             "history_after": history_after,
             "history_bag": history_bag,
             "train_test_data": train_test_data,
             "train_test_data_bagged": train_test_data_bagged,
             "train_test_data_baglabeled": train_test_data_baglabeled,
             "model": model,
             "bag_model": bag_model,
             "model_labeled": model_labeled}

pickle_out = open("output_data_dropout.pickle", "wb")
pickle.dump(data_dict, pickle_out)
"""
if model != "theor_baseline" and model != "instpred_instloss":
    print("Avg balanced accuracy before adding bag labels: {:.4f}".format(np.mean(acc_before),
                                                                          ", after: {:.4f}".format(np.mean(acc_after))))
    print('\tClass 0: {:.4f}'.format(np.mean(acc_before[:, 0])) + ' +/- {:.4f}'.format(np.std(acc_before[:, 0])) +
          ', after: {:.4f}'.format(np.mean(acc_after[:, 0])) + ' +/- {:.4f}'.format(np.std(acc_after[:, 0])))
    print('\tClass 1: {:.4f}'.format(np.mean(acc_before[:, 1])) + ' +/- {:.4f}'.format(np.std(acc_before[:, 1])) +
          ', after: {:.4f}'.format(np.mean(acc_after[:, 1])) + ' +/- {:.4f}'.format(np.std(acc_after[:, 1])))
    print("Avg loss before: ", np.mean(evals_before[:, 0]), " after: ", np.mean(evals_after[:, 0]),
          ",\n avg unbalanced acc before: ", np.mean(evals_before[:, 1]), " after: ", np.mean(evals_after[:, 1]))

    print("Avg loss on bag: ", np.mean(evals_bag[:, 0]), "+/- ", 1.96 * np.std(evals_bag[:, 0]),
          ",\n avg acc: ", np.mean(evals_bag[:, 1]), "+/- ", 1.96 * np.std(evals_bag[:, 1]),
          "(95% confidence interval)")

print("Statistics before:")
precision_before, recall_before, f1_before = \
    utils.parse_report(report_before, n_classes, print_output=True, with_stdev=True)

print("Statistics after:")
precision_after, recall_after, f1_after = \
    utils.parse_report(report_after, n_classes, print_output=True, with_stdev=True)

print("Bag model categorical accuracy: ", bagacc_max, "for_model", model)
print("Avg instance model improvement:", 100 * (np.mean(np.divide(precision_after[:, 0], precision_before[:, 0])) - 1),
      "%")
