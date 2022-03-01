from math import log2, pow
from compressors.abc_fcm_compressor import ABCFCMCompressor
from utils.contextline import ContextLine
from pympler import asizeof
from utils.utils import check_key_in_dict


class XAFCM(ABCFCMCompressor):

    def __init__(self, alphab_size, k, d, alpha="auto", p=0.9):  # , participante_numero=-1):
        # TODO refactor to avoid word_size
        self.word_size = None
        self.number_of_bits = 0.0
        self.d = d
        self.k = k
        # self.participante_numero = participante_numero

        if alpha == "auto" or alpha is None:
            self.alpha = 1.1
            prob = 0

            # while prob < p:
            while prob < pow(p, self.d):
                self.alpha /= 1.1
                prob = (1 + self.alpha) / (1 + self.alpha * pow(alphab_size, self.d))
            # print("auto alpha = %e" % self.alpha)
        # if alpha is provided
        else:
            self.alpha = alpha

        self.alphabet_size = alphab_size
        self.model_learned = dict()
        self.list_of_bits_per_symbol = []
        self._default_lidstone = self.alpha / (self.alpha * pow(self.alphabet_size, self.d))
        self._default_lidstone_part1 = self.alpha
        self._default_lidstone_part2 = self.alpha * pow(self.alphabet_size, self.d)

    def _reset_model(self):
        self.model_learned = dict()

    def _reset_number_of_bits(self):
        self.number_of_bits = 0

    # TODO implement me
    def measure_complexity(self, np_string):
        tmp_word_size = self.word_size
        if self.word_size is None:
            tmp_word_size = len(np_string)
        # TODO apagar isto..teste
        tmp_word_size = len(np_string)

        self._reset_model()
        self._reset_number_of_bits()
        self.list_of_bits_per_symbol = []

        aux_list_k = list(reversed(range(1, self.d + 1)))
        aux_list_l = list(reversed(range(self.d + 1, self.d + self.k + 1)))

        # print("np_string = ", np_string)
        # print("tmp_word_size = ", tmp_word_size)

        for curr_word_start_index in range(0, len(np_string), tmp_word_size):
            word = np_string[curr_word_start_index:curr_word_start_index + tmp_word_size]

            # print("word = %s" % word)
            for i in range(0, len(word)):

                curr_string_for_l = ""
                for curr_l in aux_list_l:
                    curr_string_for_l += word[(i - curr_l) % len(word)]

                curr_string_for_k = ""
                for curr_k in aux_list_k:
                    curr_string_for_k += word[(i - curr_k) % len(word)]

                # print(curr_string)
                prob = self.lidstone_estimate_probability_for_symbol(curr_string_for_l, curr_string_for_k)
                tmp_bits_needed = - log2(prob)
                self.list_of_bits_per_symbol.append(tmp_bits_needed)
                self.number_of_bits += tmp_bits_needed

                if not check_key_in_dict(curr_string_for_l, self.model_learned):
                    default_context_line = ContextLine(context_word=curr_string_for_l)
                    self.model_learned[curr_string_for_l] = default_context_line

                self.model_learned[curr_string_for_l].increment_symbol(curr_string_for_k)

        return self.list_of_bits_per_symbol, self.number_of_bits

    def measure_complexity_of_text_file(self, file):
        list_bps, bits = self.compress_text_file(file, based_on_model=False)
        return list_bps, bits

    def learn_models_from_string(self, np_string):
        tmp_word_size = self.word_size
        if self.word_size is None:
            tmp_word_size = len(np_string)

        # TODO check this - remove last characters in order to be multiple
        # np_string = np_string[:-(len(np_string) % tmp_word_size)]
        # assert(len(np_string) % tmp_word_size == 0)

        self._reset_model()

        aux_list_k = list(reversed(range(1, self.d + 1)))
        aux_list_l = list(reversed(range(self.d + 1, self.d + self.k + 1)))

        for curr_word_start_index in range(0, len(np_string), tmp_word_size):
            word = np_string[curr_word_start_index:curr_word_start_index + tmp_word_size]

            # print("word = %s" % word)
            for i in range(0, len(word)):

                curr_string_for_l = ""
                for curr_l in aux_list_l:
                    curr_string_for_l += word[(i - curr_l) % len(word)]

                curr_string_for_k = ""
                for curr_k in aux_list_k:
                    curr_string_for_k += word[(i - curr_k) % len(word)]

                # print(curr_string)

                if not check_key_in_dict(curr_string_for_l, self.model_learned):
                    default_context_line = ContextLine(context_word=curr_string_for_l)
                    self.model_learned[curr_string_for_l] = default_context_line

                self.model_learned[curr_string_for_l].increment_symbol(curr_string_for_k)

    def print_models_learned(self):
        print("Model learned:")
        from operator import itemgetter
        for item in sorted(self.model_learned.items(), key=itemgetter(1)):
            print(item[1])

    def get_memory_size_used_mbytes(self):
        mem_used_bytes = asizeof.asizeof(self.model_learned)
        mem_used_mbytes = mem_used_bytes / (1024 * 1024)
        return mem_used_mbytes

    # TODO change this...
    def print_memory_size_used(self):
        pass

    def print_memory_size_used_mbytes(self):
        print("RAM used: %.2fMB" % self.get_memory_size_used_mbytes())

    def print_details_of_models_learned(self):
        different_contexts_found = self.model_learned.keys()
        number_of_different_contexts_found = len(different_contexts_found)
        print("Found %s different combinations of contexts for k = %s." % (number_of_different_contexts_found, self.k))

    def lidstone_probability_part1(self, current_context_word, symbol):
        try:
            model_line = self.model_learned[current_context_word]
        # in case this word never appeared in the reference model
        except KeyError as e:
            return self._default_lidstone_part1

        try:
            tmp = model_line.cols[symbol]
        # in case this symbol never appeared for this specific word
        except KeyError as e:
            tmp = 0

        return tmp + self.alpha

    def lidstone_probability_part2(self, current_context_word, symbol):
        try:
            model_line = self.model_learned[current_context_word]
        # in case this word never appeared in the reference model
        except KeyError as e:
            return self._default_lidstone_part2

        return model_line.cols['total'] + pow(self.alphabet_size, self.d) * self.alpha

    def lidstone_estimate_probability_for_symbol(self, current_context_word, symbol):
        try:
            model_line = self.model_learned[current_context_word]
        # in case this word never appeared in the reference model
        except KeyError as e:
            return self._default_lidstone

        try:
            tmp = model_line.cols[symbol]
        # in case this symbol never appeared for this specific word
        except KeyError as e:
            tmp = 0

        return (tmp + self.alpha) / \
               (model_line.cols['total'] + pow(self.alphabet_size, self.d) * self.alpha)

    def compress_text_file(self, path_text_file, based_on_model=True):
        self._reset_number_of_bits()

        data = ""
        with open(path_text_file, "r") as my_file:
            data += my_file.read()

        data = data.replace("\n", "")

        if based_on_model:
            return self.compress_string_based_on_models(data.upper())
        # compress itself
        else:
            return self.measure_complexity(data.upper())

    def compress_string_based_on_models(self, string_to_compress):

        # compress based on self.model_learned
        self._reset_number_of_bits()
        self.list_of_bits_per_symbol = []

        tmp_word_size = self.word_size
        if self.word_size is None:
            tmp_word_size = len(string_to_compress)

        # TODO check this - remove last characters in order to be multiple
        # string_to_compress = string_to_compress[:-(len(string_to_compress) % tmp_word_size)]

        # assert(len(string_to_compress) % tmp_word_size == 0)
        assert (self.model_learned != dict())

        aux_list_next_seq = list(reversed(range(0, self.d)))
        aux_list_current_context_k = list(reversed(range(self.d, self.d + self.k)))

        for curr_word_start_index in range(0, len(string_to_compress), tmp_word_size):
            word_to_process = string_to_compress[curr_word_start_index:curr_word_start_index + tmp_word_size]

            # init curr_string
            for i in range(0, len(word_to_process), self.d):
                next_sequence = ""
                current_context_k = ""

                for i_tmp in aux_list_next_seq:
                    next_sequence += word_to_process[(i - i_tmp) % len(word_to_process)]

                for i_tmp in aux_list_current_context_k:
                    current_context_k += word_to_process[(i - i_tmp) % len(word_to_process)]

                prob = self.lidstone_estimate_probability_for_symbol(current_context_k, next_sequence)
                tmp_bits_needed = - log2(prob)
                self.list_of_bits_per_symbol.append(tmp_bits_needed)
                self.number_of_bits += tmp_bits_needed

        return self.list_of_bits_per_symbol, self.number_of_bits


# def main():
#     str_test = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
#     xafcm = XAFCM(alphab_size=3, k=2, d=1, alpha="auto")
#     xafcm.learn_models_from_string(str_test)
#     tmp1 = xafcm.compress_string_based_on_models(str_test)
#     print(tmp1)
#
#     print("Complexity v1:")
#     tmp2 = xafcm.measure_complexity(str_test)
#     print(tmp2)
#
#     print("Complexity v2:")
#     tmp3 = xafcm.measure_complexity_of_text_file("C:\\Users\\jifup\\OneDrive - Universidade de Aveiro\\workspace\\fcmclassifier\\teste.txt")
#     print(tmp3)
#
#
# if __name__ == "__main__":
#     main()
