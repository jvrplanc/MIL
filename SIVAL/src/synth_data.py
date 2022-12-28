import threading

import numpy as np
import pandas as pd
import sdv
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table import QualityReport

lock = threading.Lock()

"""
A lot of hardcoded stuff in here. Sorry
"""


def metadata():
    metadata_dict = {
        "fields": {
            "class": {"type": "id", "subtype": "string"},
            "bag": {"type": "id", "subtype": "string"},
            "instance_id": {"type": "id", "subtype": "integer"},
            "1": {"type": "numerical", "subtype": "float"},
            "2": {"type": "numerical", "subtype": "float"},
            "3": {"type": "numerical", "subtype": "float"},
            "4": {"type": "numerical", "subtype": "float"},
            "5": {"type": "numerical", "subtype": "float"},
            "6": {"type": "numerical", "subtype": "float"},
            "7": {"type": "numerical", "subtype": "float"},
            "8": {"type": "numerical", "subtype": "float"},
            "9": {"type": "numerical", "subtype": "float"},
            "10": {"type": "numerical", "subtype": "float"},
            "11": {"type": "numerical", "subtype": "float"},
            "12": {"type": "numerical", "subtype": "float"},
            "13": {"type": "numerical", "subtype": "float"},
            "14": {"type": "numerical", "subtype": "float"},
            "15": {"type": "numerical", "subtype": "float"},
            "16": {"type": "numerical", "subtype": "float"},
            "17": {"type": "numerical", "subtype": "float"},
            "18": {"type": "numerical", "subtype": "float"},
            "19": {"type": "numerical", "subtype": "float"},
            "20": {"type": "numerical", "subtype": "float"},
            "21": {"type": "numerical", "subtype": "float"},
            "22": {"type": "numerical", "subtype": "float"},
            "23": {"type": "numerical", "subtype": "float"},
            "24": {"type": "numerical", "subtype": "float"},
            "25": {"type": "numerical", "subtype": "float"},
            "26": {"type": "numerical", "subtype": "float"},
            "27": {"type": "numerical", "subtype": "float"},
            "28": {"type": "numerical", "subtype": "float"},
            "29": {"type": "numerical", "subtype": "float"},
            "30": {"type": "numerical", "subtype": "float"},
            "instance_label": {"type": "categorical"}
        }
    }
    return metadata_dict


def metadata_report():
    metadata_dict = {
        "fields": {
            "1": {"type": "numerical", "subtype": "float"},
            "2": {"type": "numerical", "subtype": "float"},
            "3": {"type": "numerical", "subtype": "float"},
            "4": {"type": "numerical", "subtype": "float"},
            "5": {"type": "numerical", "subtype": "float"},
            "6": {"type": "numerical", "subtype": "float"},
            "7": {"type": "numerical", "subtype": "float"},
            "8": {"type": "numerical", "subtype": "float"},
            "9": {"type": "numerical", "subtype": "float"},
            "10": {"type": "numerical", "subtype": "float"},
            "11": {"type": "numerical", "subtype": "float"},
            "12": {"type": "numerical", "subtype": "float"},
            "13": {"type": "numerical", "subtype": "float"},
            "14": {"type": "numerical", "subtype": "float"},
            "15": {"type": "numerical", "subtype": "float"},
            "16": {"type": "numerical", "subtype": "float"},
            "17": {"type": "numerical", "subtype": "float"},
            "18": {"type": "numerical", "subtype": "float"},
            "19": {"type": "numerical", "subtype": "float"},
            "20": {"type": "numerical", "subtype": "float"},
            "21": {"type": "numerical", "subtype": "float"},
            "22": {"type": "numerical", "subtype": "float"},
            "23": {"type": "numerical", "subtype": "float"},
            "24": {"type": "numerical", "subtype": "float"},
            "25": {"type": "numerical", "subtype": "float"},
            "26": {"type": "numerical", "subtype": "float"},
            "27": {"type": "numerical", "subtype": "float"},
            "28": {"type": "numerical", "subtype": "float"},
            "29": {"type": "numerical", "subtype": "float"},
            "30": {"type": "numerical", "subtype": "float"}
        }
    }
    return metadata_dict


def set_dtypes(df, start, stop):
    for i in range(start, stop + 1):
        i_str = str(i)
        df[i_str] = pd.to_numeric(df[i_str], downcast='float')  # Assuming column names are str(i) here
    return df


def constraints():
    c_f1 = sdv.constraints.ScalarRange(column_name="1", low_value=0, high_value=255, strict_boundaries=False)
    c_f2 = sdv.constraints.ScalarRange(column_name="2", low_value=0, high_value=255, strict_boundaries=False)
    c_f3 = sdv.constraints.ScalarRange(column_name="3", low_value=0, high_value=255, strict_boundaries=False)
    c_f4 = sdv.constraints.Positive(column_name="4")
    c_f5 = sdv.constraints.Positive(column_name="5")
    c_f6 = sdv.constraints.Positive(column_name="6")

    constraints = [c_f1, c_f2, c_f3, c_f4, c_f5, c_f6]

    return constraints


def augment_dataset(input_data, model=None, verbose=False):
    columns = ["class", "bag", "instance_id"]
    columns.extend(np.ndarray.tolist(np.arange(1, 31).astype('str')))
    columns.append("instance_label")

    input_df = pd.DataFrame(input_data, columns=columns)
    input_df.insert(loc=2, column="is_synthetic", value=0)
    input_df = set_dtypes(input_df, 1, 30)  # str(int) = column names

    if model == "TVAE":
        model = sdv.tabular.TVAE()
    elif model == "CTGAN":
        model = sdv.tabular.CTGAN()
    elif model == "GaussianCopula":
        model = sdv.tabular.GaussianCopula()
    elif model == "CopulaGAN":
        model = sdv.tabular.CopulaGAN()
    else:
        model = sdv.tabular.CTGAN()

    lock.acquire()  # samples are written to a temp file with fixed name... Weird stuff also happens when not locking
    # fit method so probably race condition

    model.fit(input_df)

    # fill the synthetic input_df with the same instance labels, such that every bag has a positive instance. A better way
    # to do this would be to randomly shuffle but ensuring at least one positive instance per bag...
    bag_names = ["s_" + instance[1] for instance in input_data]

    # fill instance label column to maintain same distribution, then create the data around that
    conditions = pd.DataFrame({"instance_label": input_data[:, -1]})
    sample = model.sample_remaining_columns(conditions)

    lock.release()

    # assign dummy bag names, flag as synthetic
    sample["bag"] = bag_names
    sample["is_synthetic"] = 1

    if verbose:
        metadata_dict = metadata_report()
        report = QualityReport()
        report.generate(input_df.T[4:-1].T.astype(np.float32), sample.T[4:-1].T.astype(np.float32), metadata_dict)
        report = DiagnosticReport()
        report.generate(input_df.T[4:-1].T.astype(np.float32), sample.T[4:-1].T.astype(np.float32), metadata_dict)

    augmented_data = pd.concat([input_df, sample])
    augmented_data = augmented_data.to_numpy(dtype='O')  # for compatibility with the rest of the code

    return augmented_data
