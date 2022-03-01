import numpy as np
import pandas as pd


class DiffQuantizer:
    def __init__(self, alphabet_size, average_over=1, breakpoints=None, use_diffs=True):

        self.alphabet_size = alphabet_size
        self.average_over = average_over
        self.breakpoints = breakpoints
        self.use_diffs = use_diffs

    def preprocess(self, tmp):
        if self.average_over != 1:
            tmp = self._average_over_n(tmp, self.average_over)

        if self.use_diffs:
            tmp = self._diff_signal(tmp)

        return tmp

    def perform_quantization(self, tmp, breakpoints=None):
        self.breakpoints = breakpoints

        tmp = self.preprocess(tmp)
        result = self._quantize_with_breakpoints(tmp)
        return result

    def learn_breakpoints(self, arr):
        res = self.preprocess(arr)

        sorted_array = np.sort(res)

        length = len(sorted_array)

        probs = [1 / self.alphabet_size for _ in range(self.alphabet_size)]
        cum_sum_breakpoints = [int(sum(probs[0:i + 1]) * length - 1) for i in range(len(probs))]
        cum_sum_breakpoint_values = sorted_array[cum_sum_breakpoints]

        # "hack" to avoid some values above last breakpoint
        # TODO change to something more pretty :) (but sys.maxsize doesn't work)
        cum_sum_breakpoint_values[-1] = 1e+100

        self.breakpoints = cum_sum_breakpoint_values
        return cum_sum_breakpoint_values

    # vectorized use
    @staticmethod
    def _breakpoint_to_letter(float_num, breakpoints):
        int_val = next((breakpoints.index(obj) for obj in breakpoints if float_num < obj))

        # A + int_val
        return chr(65 + int_val)

    def _quantize_with_breakpoints(self, tmp):
        breakpoints = self.breakpoints
        vect_breakpoint_to_letter = np.vectorize(self._breakpoint_to_letter, excluded=['breakpoints'])
        tmp = vect_breakpoint_to_letter(tmp, breakpoints=list(breakpoints))
        return tmp

    @staticmethod
    def read_csv_file(input_file):
        tmp_file_content = pd.read_csv(input_file, sep="\n", header=None, dtype=np.float64)[0]
        return np.array(tmp_file_content)

    @staticmethod
    def _average_over_n(tmp, n):
        return np.array([np.average(tmp[i:i + n]) for i in range(0, len(tmp), n)])

    @staticmethod
    def _diff_signal(tmp):
        res = np.diff(tmp)
        return np.insert(res, 0, 0.0)

    def set_breakpoints_from_file(self, filepath):
        self.breakpoints = np.loadtxt(filepath)

    @staticmethod
    def get_breakpoints_from_file(filepath):
        return list(np.loadtxt(filepath, dtype='f4'))

