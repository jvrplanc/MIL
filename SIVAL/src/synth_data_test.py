import os

import numpy as np

import data_proc
import fit_model_inst
import models
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU faster than GPU on my machine

"""
If you run this, you'll need to change some hard-coded 3:'s to 4:'s in the other files as well as update omit_first_cols
to 4
"""

SIVAL_data_dir = "../input/amil-sival/"

sd_models = ["GaussianCopula", "CopulaGAN", "TVAE", "CTGAN"]
precision = dict.fromkeys(sd_models)
recall = dict.fromkeys(sd_models)
f1 = dict.fromkeys(sd_models)

for sd_model in sd_models:
    data_flat = data_proc.get_data(SIVAL_data_dir, ".data", synthetic=True, model=sd_model)
    bag_labels, data_bagged = utils.split_into_bags(data_flat, bag_id_column=1)

    n_splits = 10
    omit_first_cols = 4
    max_epochs = 250

    synth_column = np.array([bag[0, 2] for bag in data_bagged])

    data_bagged_im, data_bagged_bm, bag_labels_im, bag_labels_bm = \
        utils.split_real_synth_bags(synth_column, data_bagged, bag_labels)

    n_classes = np.max(data_flat[:, -1])

    train_test_data = data_proc.split_proc_inst_wrapper(n_splits,
                                                        np.vstack(data_bagged_im),
                                                        omit_first_cols=omit_first_cols,
                                                        validation_split=0.2,
                                                        verbose=True)

    n_feat = train_test_data[0].X_train.shape[1] - omit_first_cols

    inst_model = models.inst_model(n_feat, n_classes)

    history_before, evals_before, acc_before = fit_model_inst.fit_and_test(n_splits=n_splits,
                                                                           model=inst_model,
                                                                           data=train_test_data,
                                                                           omit_first_cols=omit_first_cols,
                                                                           epochs=max_epochs,
                                                                           verbose=True)
    print("Finished fitting without bag labels.\n Statistics:")
    report_before = utils.gather_stats(history_before, train_test_data, n_splits, omit_first_cols)

    precision_before, recall_before, f1_before = \
        utils.parse_report(report_before, n_classes, print_output=True, with_stdev=True)

    precision[sd_model] = precision_before
    recall[sd_model] = recall_before
    f1[sd_model] = f1_before
