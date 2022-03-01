from sklearn.datasets import load_files
from compressors.fcm_extended_alphabet import XAFCM
import numpy as np
import pandas as pd
from math import log2
import multiprocessing
from pathlib import Path
from compressors.fcm_mixtures import FCMMixtures


def get_label_number(target_names, target):
    return target_names.index(target)


def get_label_name(target_names, number):
    return target_names[number]


def load_files_and_assign_variables(path, shuffle=False):
    # READ FILES
    print("Loading files from", path)
    x = load_files(path, shuffle=shuffle, encoding='ascii')
    # print("Found {0} files/folders:\n{1}\n".format(len(x.filenames), x.filenames))
    print("Found {0} files/folders\n".format(len(x.filenames)))
    return x.data, x.target, x.target_names, x.filenames


def compute_nrc_for_X_test_i_for_model(i):
    global X_test
    global fcm
    x = X_test[i]
    _, bits = fcm.compress_string_based_on_models(x)
    nrc = bits / float(len(x) * log2(fcm.alphabet_size))
    return i, nrc


# defining global variables (hack to deal with multiprocessing)
X_test = None
fcm = None
df_nrcs = None


def main(input_dir, alphabet, k, d, output_csv=None, threads=1):
    global df_nrcs
    df_nrcs = None
    input_dir = Path(input_dir)

    # CONSTANTS
    if threads == -1:
        threads = multiprocessing.cpu_count()
    else:
        threads = threads

    # READ FILES
    X_train, y_train, train_target_names, train_filenames = load_files_and_assign_variables(input_dir / "train")
    print(train_target_names)
    print()

    # CLASSIFY BY COMPRESSION

    for current_model in train_target_names:
        global fcm
        global X_test

        fcm = XAFCM(alphab_size=alphabet, k=k, d=d, alpha="auto")
        # fcm = FCMMixtures(alphab_size=alphabet,
        #                   # model_orders=[5, 6, 7, 8, 9],
        #                   model_orders=[5, 8, 11, 14, 17],
        #                   # model_orders=[11, 14, 17, 20, 25],
        #                   model_alphas=[0.001, 0.001, 0.001, 0.001, 0.001],
        #                   word_size=None,
        #                   forgetting_factor=0.2)

        print("Learning model for", current_model)
        indexes_for_label = [i for i, j in enumerate(y_train) if
                             j == get_label_number(train_target_names, current_model)]

        curr_training_data = [''.join(X_train[i]) for i in indexes_for_label]
        curr_training_data = ''.join(curr_training_data)
        fcm.learn_models_from_string(curr_training_data)

        X_test, y_test, test_target_names, test_filenames = \
            load_files_and_assign_variables(input_dir / "test" / current_model)

        print("Computing NRC measures")
        # print("current_model = ", current_model)

        with multiprocessing.Pool(threads) as p:
            i_nrc_tuples_list = p.map(compute_nrc_for_X_test_i_for_model, range(len(X_test)))

        def update_dataframe_with_tuple_results(tuples):
            indexes = [i for i in range(len(X_test))]
            columns = train_target_names + ['Target']

            # if not defined yet
            global df_nrcs
            if df_nrcs is None:
                df_nrcs = pd.DataFrame(index=indexes, columns=columns, dtype=float)
                df_nrcs[['Target']] = df_nrcs[['Target']].astype(str)

            for t in tuples:
                i = t[0]
                nrc = t[1]
                df_nrcs.at[i, current_model] = nrc
                real_label = y_test[i]

                df_nrcs.at[i, 'Target'] = get_label_name(train_target_names, real_label)

        update_dataframe_with_tuple_results(i_nrc_tuples_list)
        del fcm

    # ----------------------------------------------------------
    #      GET CLASS WITH LOWER NRC AND COMPUTE ACCURACY
    # ----------------------------------------------------------
    print("columns = ", df_nrcs.columns)
    df_nrcs['Guess'] = df_nrcs.drop('Target', axis=1).idxmin(axis=1)

    cols_to_ignore = ['Target', 'Guess']
    
    if output_csv is not None:
        df_nrcs.to_csv(output_csv)
    return df_nrcs


if __name__ == "__main__":
    main()
