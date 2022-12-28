import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import IterativeStratification


def split_real_synth_bags(split_col, input_data, bag_labels, *args, **kwargs):
    # split along real/synth (= split_col) but also bag label
    bag_labels = bag_labels.flatten().astype('int32')  # just making sure
    multilabel_col = np.array([(split_col[i], bag_labels[i]) for i, label in enumerate(bag_labels.flatten())])  # or np.vstack.T?

    strat = IterativeStratification(n_splits=2, order=2)
    splits = strat.split(input_data, multilabel_col)

    output_data = []
    for i, (split1, split2) in enumerate(splits):
        output_data.append(input_data[split1])
        output_data.append(input_data[split2])
        output_data.append(bag_labels[split1])
        output_data.append(bag_labels[split2])
        for arg in args:
            output_data.append(arg[split1])
            output_data.append(arg[split2])
        break

    """
    i = 1
    data = output_data[i]
    synthspread = np.bincount([bag[0, 2] for bag in data])
    bagspread = np.bincount([np.max(bag[:, -1]) for bag in data])
    """
    return output_data

def split_dataset_bags(input_data, bag_labels, *args, **kwargs):
    strat = IterativeStratification(n_splits=2, order=1)
    splits = strat.split(input_data, bag_labels)

    output_data = []
    for i, (split1, split2) in enumerate(splits):
        output_data.append(input_data[split1])
        output_data.append(input_data[split2])
        output_data.append(bag_labels[split1])
        output_data.append(bag_labels[split2])
        for arg in args:
            output_data.append(arg[split1])
            output_data.append(arg[split2])
        break

    """
    i = 1
    data = output_data[i]
    synthspread = np.bincount([bag[0, 2] for bag in data])
    bagspread = np.bincount([np.max(bag[:, -1]) for bag in data])
    """
    return output_data


def tensorify(data, omit_first_cols=3):
    ragged_tensor = np.asarray([bag[:, omit_first_cols:-1].astype('float32') for bag in data], dtype='object')
    ragged_tensor = tf.ragged.constant(ragged_tensor, ragged_rank=1)

    return ragged_tensor


def tensorify_and_expand(data, omit_first_cols=3):
    ragged_tensor = np.asarray([bag[:, omit_first_cols:-1].astype('float32') for bag in data], dtype='object')
    ragged_tensor = np.asarray([np.expand_dims(row, -1) for row in ragged_tensor], dtype='object')
    ragged_tensor = tf.ragged.constant(ragged_tensor, ragged_rank=1)
    tensor = ragged_tensor.to_tensor()

    return tensor


def gather_stats(history, dataset, n_splits, omit_first_cols=3):
    report = []
    for i in range(n_splits):
        y_pred = history[i].model.predict(dataset[i].X_test[:, omit_first_cols:].astype('float32'))
        y_pred = np.argmax(y_pred, axis=1)
        y_test = dataset[i].y_test.astype('int64').flatten()
        #print(np.sum(np.equal(y_pred, y_test)))
        report.append(classification_report(y_test, y_pred, output_dict=True))

    return report


def parse_report(input_data, n_classes, print_output=False, with_stdev=False):
    precision_all = np.zeros(shape=(len(input_data), n_classes + 1))
    recall_all = np.zeros(shape=(len(input_data), n_classes + 1))
    f1_all = np.zeros(shape=(len(input_data), n_classes + 1))

    # returns ndarray of n_splits rows and n_classes columns with values
    for i, split in enumerate(input_data):
        for j, class_label in enumerate(split):
            if class_label.isnumeric():
                precision_all[i, j] = split[class_label]["precision"]
                recall_all[i, j] = split[class_label]["recall"]
                f1_all[i, j] = split[class_label]["f1-score"]

    precision = np.vstack(
        [np.nanmean(precision_all, axis=0), np.nanstd(precision_all, axis=0)]).transpose()  # avg and stdev
    recall = np.vstack([np.nanmean(recall_all, axis=0), np.nanstd(recall_all, axis=0)]).transpose()
    f1 = np.vstack([np.nanmean(f1_all, axis=0), np.nanstd(f1_all, axis=0)]).transpose()

    if print_output:
        if with_stdev:
            print("Mean_precision stdev_precision mean_recall stdev_recall f1 stdev_f1")
        else:
            print("Mean_precision mean_recall f1")
        for i in range(precision.shape[0]):
            mean_prec = precision[i, 0]
            std_prec = precision[i, 1]
            mean_rec = recall[i, 0]
            std_rec = recall[i, 1]
            mean_f1 = f1[i, 0]
            std_f1 = f1[i, 1]

            if with_stdev:
                print(i, mean_prec, std_prec, mean_rec, std_rec, mean_f1, std_f1)
            else:
                print(i, mean_prec, mean_rec, f1)

    return precision, recall, f1


def split_into_bags(input_array, bag_id_column):
    unique, s = np.unique(input_array[:, bag_id_column], return_index=True)  # returns positions of unique image data
    indices = np.argsort(s)
    unique, s = unique[indices], s[indices]

    split_array = np.asarray(np.split(input_array, s[1:]), dtype='object')
    bag_labels = np.vstack([np.max(bag[:, -1]) for bag in split_array])

    # keep the arrays as flat as possible because nesting is confusing enough as it is
    return bag_labels, split_array
