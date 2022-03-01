import os
import subprocess
import matplotlib as mpl
import shutil
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score
from datetime import datetime
from time import sleep

def check_key_in_dict(key, some_dict):
    try:
        tmp = some_dict[key]
        return True
    except KeyError as e:
        return False


def if_dir_does_not_exist_create(dir, delete_contents=False):
    if delete_contents and os.path.exists(dir):
        shutil.rmtree(dir)

    if not os.path.exists(dir):
        os.makedirs(dir)


def get_all_files(root_dir):
    result = []
    for path, _, files in os.walk(root_dir):
        for name in files:
            result.append(os.path.join(path, name))
    return result


def add_to_df(dataf, row, column, value):
    if column not in dataf:
        dataf[column] = 0.0

    dataf.loc[row][column] = value


def add_mean_and_std_to_dataframe(dataf, on_cols=True, on_rows=True):
    if on_rows:
        # add mean COLUMN
        dataf['mean'] = dataf.mean(numeric_only=True, axis=1)
        dataf['std'] = dataf.std(numeric_only=True, axis=1)

    if on_cols:
        # add mean and std ROWS
        dataf.loc['mean'] = dataf.mean(numeric_only=True, axis=0)
        dataf.loc['std'] = dataf.std(numeric_only=True, axis=0)

    if on_cols and on_rows:
        # set these cells to NAN
        dataf['mean']['mean'] = np.NAN
        dataf['mean']['std'] = np.NAN
        dataf['std']['mean'] = np.NAN
        dataf['std']['std'] = np.NAN
    return dataf


def add_min_and_max_to_dataframe(dataf, on_cols=True, on_rows=True):
    if on_rows:
        # add mean COLUMN
        dataf['min'] = dataf.min(numeric_only=True, axis=1)
        dataf['max'] = dataf.max(numeric_only=True, axis=1)

    if on_cols:
        # add mean and std ROWS
        dataf.loc['min'] = dataf.min(numeric_only=True, axis=0)
        dataf.loc['max'] = dataf.max(numeric_only=True, axis=0)

    if on_cols and on_rows:
        # set these cells to NAN
        dataf['min']['min'] = np.NAN
        dataf['min']['max'] = np.NAN
        dataf['max']['min'] = np.NAN
        dataf['max']['max'] = np.NAN
    return dataf


def resize_image_to_square(im_pth, desired_size=512):
    im = Image.open(im_pth)
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))
    new_im.save(im_pth)


def save_spectogram(array, rate, output_file):
    global file_name
    f, t, Sxx = signal.spectrogram(array, fs=rate)
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0)
    plt.subplots_adjust(top=1)
    plt.subplots_adjust(right=1)
    plt.subplots_adjust(left=0)
    ax.pcolormesh(t, f, Sxx, norm=mpl.colors.LogNorm(vmin=Sxx.min(), vmax=Sxx.max()), cmap='inferno')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    my_dpi = 96
    print("Saving fig", output_file)
    plt.savefig(output_file, dpi=my_dpi)


def get_name_without_extension(path):
    return path.name.replace(path.suffix, '')


def compute_and_print_metrics(y_true, y_pred, average='macro'):
    f1_original = f1_score(y_true, y_pred, average=average)
    accuracy_original = accuracy_score(y_true, y_pred)
    precision_original = precision_score(y_true, y_pred, average=average)
    print("f1 score = {0:.2f}".format(f1_original))
    print("accuracy score = {0:.2f}".format(accuracy_original))
    print("precision score = {0:.2f}".format(precision_original))


def file_size_bytes(fname):
    tmp = run_terminal_cmd("ls -l {}".format(fname), output=True)
    bytes = int(str(tmp).split(' ')[4])
    print("{0} bytes ({1} bits; {2:.2f}KBytes; {3:.2f}MBytes"
          .format(bytes, bytes * 8, bytes / 1024, bytes / 1024 / 1024))
    return bytes


def run_terminal_cmd(cmd, output=False, verbose=False, decode=False):
    if not output:
        cmd += "&>/dev/null"

    if verbose:
        print("running cmd:", cmd)

    if output:
        res = subprocess.check_output(cmd, shell=True)

        if decode:
            res = res.decode('utf-8')

        return res
    else:
        os.system(cmd)


def get_current_datetime():
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    return dt_string

