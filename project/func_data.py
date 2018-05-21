import copy
import numpy as np
from pandas import DataFrame as df


def load_data(data_dir="data.csv", n_train=40, shuffle=True):
    # DEFINITION
    # ==========
    data_dir = "data.csv"

    # MAKE INPUT DATA
    # ===============

    # Load data
    # ---------

    data_raw = df.from_csv(data_dir, index_col=0)

    # Shuffle data
    # ------------

    if shuffle:
        data_raw = data_raw.sample(frac=1).reset_index(drop=True)

    # Data sanitation
    # ---------------

    # convert all headers to lower case
    data_raw.columns = [i.lower().replace(' ', '_') for i in data_raw.columns]

    # convert all data to lower case
    for i in data_raw:
        _data_type = data_raw[i].dtype
        if not np.issubdtype(_data_type, np.number):
            data_raw[i] = data_raw[i].str.lower()

    # remove post incident columns
    # data_raw.pop('ignition')
    # data_raw.pop('ignition_floor')
    # data_raw.pop('wind')
    # data_raw.pop('falling_material')
    # data_raw.pop('under_construction')

    # MAKE LABEL/CLASS
    # ================

    # Obtain Fatality and Injury
    # ---------------------------------------------------

    # categorise label column, i.e. 'hazard_classification'
    hazard_classification = copy.copy(data_raw['fatalities'])
    hazard_classification = hazard_classification.astype(int)

    # delete Fatality and Injury from original data sheet
    fatalities = data_raw.pop("fatalities")
    injuries = data_raw.pop("injuries")

    # classification Fatality and Injury buckets
    risk_class_fatality = [1, 5, 10, 20, 50, 1e10]
    risk_class_injuries = [1, 5, 10, 20, 50, 1e10]

    # calculation
    for i, v_fatalities in enumerate(list(fatalities)):

        v_injuries = injuries.iloc[i]

        if v_fatalities > 0:
            for i_ in np.arange(1, len(risk_class_fatality)):
                bound_1 = risk_class_fatality[i_ - 1]
                bound_2 = risk_class_fatality[i_]
                if bound_1 <= v_fatalities < bound_2:
                    hazard_classification.iloc[i] = i_ + len(risk_class_injuries)
                    break
        elif v_injuries > 0:
            for i_ in np.arange(1, len(risk_class_injuries)):
                bound_1 = risk_class_injuries[i_ - 1]
                bound_2 = risk_class_injuries[i_]
                if bound_1 <= v_injuries < bound_2:
                    hazard_classification.iloc[i] = i_
                    break
        elif v_fatalities == 0 and v_injuries == 0:
            hazard_classification.iloc[i] = 0
            continue

        else:
            print("train feature category not recognised: {}, {}".format(i, v_fatalities))

    data_raw['hazard_classification'] = hazard_classification

    # SAVE SANITISED DATA
    # ===================

    # this is for testing purpose, to inspect the clean data and produced classes
    data_raw.to_csv('data_clean.csv')

    # SELECT TRAINING AND TESTING DATASET
    # ===================================

    # 'n_train' is the defined number for training data items. first 'n_train' rows in 'data_raw' are selected for model
    # training purpose. what is left are used for model testing purpose

    # at this point, rows in 'data_raw' are shuffled, therefore, the selected training and testing data are shuffled

    train_xy = data_raw[:n_train]
    test_xy = data_raw[n_train:]

    # index numbering for training and testing DataFrame are sorted. this should not affect anything but for debugging
    # purpose

    train_xy.reset_index(drop=True, inplace=True)
    test_xy.reset_index(drop=True, inplace=True)

    # separate feature columns and label column for training and testing data

    train_x, train_y = train_xy, train_xy.pop('hazard_classification')
    test_x, test_y = test_xy, test_xy.pop('hazard_classification')

    # DEBUG
    # =====

    # str_labels = list(all_label.drop_duplicates())
    # str_labels.sort()
    # print(str_labels)

    return (train_x, train_y), (test_x, test_y)
