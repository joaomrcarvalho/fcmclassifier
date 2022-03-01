import math
import sys
import numpy as np
from compressors.abc_fcm_compressor import ABCFCMCompressor
from utils.contextline import ContextLine
from pympler import asizeof
from utils.utils import check_key_in_dict


class FCMMixtures(ABCFCMCompressor):
    def __init__(self, alphab_size, model_orders, model_alphas,
                 model_initial_weights=None, forgetting_factor=0.9,
                 word_size=None, with_neural_network=False, numbers=False):
        # not being used so far
        self.word_size = word_size

        # if using digits instead of alphabet
        self.numbers = numbers
        # BEGIN TESTS
        self.with_neural_network = with_neural_network

        # END OF TESTS

        assert (len(model_alphas) == len(model_orders))
        self.number_of_models = len(model_orders)

        self.model_orders = model_orders
        self.model_alphas = model_alphas

        self.model_weights = []
        self._init_weights(model_initial_weights)

        self.forgetting_factor = forgetting_factor
        self.alphabet_size = alphab_size

        self._symbols = ""
        self.symbols_used = ""
        self._init_symbols()

        self.models_learned = list()
        self._init_models_learned()

        self.number_of_bits = 0.0
        self.list_of_bits_per_symbol = []

        self._default_lidstones = list()
        self._init_default_lidstones()

    def get_memory_size_used_mbytes(self):
        mem_used_bytes = asizeof.asizeof(self.models_learned)
        mem_used_mbytes = mem_used_bytes / (1024 * 1024)
        return mem_used_mbytes

    # to get better performance
    def _init_default_lidstones(self):
        for it in range(len(self.model_orders)):
            self._default_lidstones.append(self.model_alphas[it] / (self.alphabet_size * self.model_alphas[it]))

    def _init_weights(self, initial_weights):
        if initial_weights is None:
            for i in range(self.number_of_models):
                self.model_weights.append(1 / float(self.number_of_models))
        else:
            self.model_weights = initial_weights

    def _init_symbols(self):
        if self.numbers:
            self.symbols_used = list(range(len(self.model_orders)))
        else:
            for letter_ord in range(ord('A'), ord('Z') + 1):
                self._symbols += chr(letter_ord)
            self.symbols_used = self._symbols[0:self.alphabet_size]

    def _init_models_learned(self):
        for x in range(self.number_of_models):
            self.models_learned.append(dict())

    def _reset_models(self):
        self.models_learned = []
        self._init_models_learned()

    def _reset_number_of_bits(self):
        self.number_of_bits = 0

    def learn_models_from_string(self, np_string):
        self._reset_models()

        for it in range(self.number_of_models):
            current_model_order = self.model_orders[it]

            for i in range(current_model_order, len(np_string)):
                curr_np_string = np_string[i - current_model_order:i]
                curr_np_string = ''.join(curr_np_string)

                if not check_key_in_dict(curr_np_string, self.models_learned[it]):
                    default_context_line = ContextLine(context_word=curr_np_string)
                    self.models_learned[it][curr_np_string] = default_context_line

                self.models_learned[it][curr_np_string].increment_symbol(np_string[i])

    def print_models_learned(self):
        print("Models learned:")
        for it in range(self.number_of_models):
            model = self.models_learned[it]
            print("Model for k = %d and alpha = %.5f" % (self.model_orders[it], self.model_alphas[it]))
            for key in model.keys():
                print(model[key])

    def print_details_of_models_learned(self):
        different_contexts_found = 0
        for model in self.models_learned:
            different_contexts_found += len(model.keys())
        print("Found %s different combinations of contexts for k = %s." % (different_contexts_found, self.model_orders))

    def print_memory_size_used(self):
        mem_used_bytes = 0
        for model in self.models_learned:
            mem_used_bytes += sys.getsizeof(model)
        mem_used_mbytes = mem_used_bytes / 1024
        mem_used_gbytes = mem_used_mbytes / 1024
        print("RAM used: %s bytes; %.2fMB; %.2fGB" % (mem_used_bytes, mem_used_mbytes, mem_used_gbytes))

    def lidstone_estimate_probability_for_symbol(self, current_context_word, symbol, it):
        try:
            current_context_word = ''.join(current_context_word)
            model_line = self.models_learned[it][current_context_word]
        # in case this word never appeared in the reference model
        except KeyError as e:
            return self._default_lidstones[it]

        try:
            tmp = model_line.cols[symbol]
        # in case this symbol never appeared for this specific word
        except KeyError as e:
            tmp = 0

        return (tmp + self.model_alphas[it]) / \
               (model_line.cols['total'] + self.alphabet_size * self.model_alphas[it])

    def _compute_model_chain(self, string):
        # self._model_chain_fcm = FCMMixtures(alphab_size=self.alphabet_size,
        #                                     model_orders=self.model_orders,
        #                                     model_alphas=self.model_alphas,
        #                                     word_size=self.word_size,
        #                                     forgetting_factor=self.forgetting_factor,
        #                                     numbers=True)

        self._model_chain_fcm = FCMMixtures(alphab_size=self.alphabet_size,
                                            model_orders=[20, 40],
                                            model_alphas=[1, 1],
                                            word_size=self.word_size,
                                            # forgetting_factor=self.forgetting_factor,
                                            numbers=True)

        assert (dict() not in self.models_learned)

        model_chain = ""
        log2_alphabet = math.log2(self.alphabet_size)
        max_k = (max(self.model_orders))

        # TODO - change this for optimal result
        for i in range(0, max_k):
            model_chain += str(0)

        # initializing lists/arrays/whatever
        curr_strings = ["" for x in range(self.number_of_models)]
        curr_probs = [0.0 for x in range(self.number_of_models)]

        # i is the position being compressed
        for i in range(max_k, len(string)):

            for it in range(self.number_of_models):
                curr_strings[it] = string[i - self.model_orders[it]:i]

            for it in range(self.number_of_models):
                curr_probs[it] = self.lidstone_estimate_probability_for_symbol(
                    curr_strings[it], string[i], it)

            best_model = np.argmax(curr_probs)
            model_chain += str(best_model)

        self._model_chain_fcm.learn_models_from_string(model_chain)
        self._model_chain_fcm.print_models_learned()
        print("MODEL_CHAIN = ", model_chain)

        return model_chain

    # TODO implement me
    def compress_string_based_on_models_and_model_chain(self, string_for_model_chain, string_to_compress):
        self._reset_number_of_bits()
        self.list_of_bits_per_symbol = []
        assert (dict() not in self.models_learned)

        self._compute_model_chain(string_for_model_chain)

        log2_alphabet = math.log2(self.alphabet_size)
        min_k = (min(self.model_orders))
        max_k = (max(self.model_orders))

        # init models_chain strings
        curr_chains = ["" for x in range(self.number_of_models)]
        for it in range(self.number_of_models):
            for i in range(self.model_orders[it]):
                curr_chains[it] += str(0)

        # print(curr_chains)

        # TODO - change this for optimal result
        for i in range(0, max_k):
            self.number_of_bits += float(log2_alphabet)
            self.list_of_bits_per_symbol.append(log2_alphabet)

        # initializing lists/arrays/whatever
        curr_strings = ["" for x in range(self.number_of_models)]
        curr_chain_probs = [0.0 for x in range(self._model_chain_fcm.number_of_models)]

        prev_model = str(0)
        # i is the position being compressed
        for i in range(max_k, len(string_to_compress)):

            for it in range(self.number_of_models):
                curr_strings[it] = string_to_compress[i - self.model_orders[it]:i]
                # TODO check

            for it_chain in range(self._model_chain_fcm.number_of_models):
                curr_chains[it_chain] = curr_chains[it_chain][1:] + str(prev_model)

            # TODO uncomment
            print(curr_chains)

            # TODO
            # for it in range(self.number_of_models):
            #     curr_chains[it] = curr_chains[it][1:] + prev_model

            for it_chain in range(self._model_chain_fcm.number_of_models):
                curr_chain_probs[it_chain] = self._model_chain_fcm.lidstone_estimate_probability_for_symbol(
                    current_context_word=curr_chains[it_chain],
                    symbol=str(it_chain),
                    it=it_chain)

            print(curr_chain_probs)
            prev_model = np.argmax(curr_chain_probs)

            # calculate bits needed
            tmp_prob = self.lidstone_estimate_probability_for_symbol(
                current_context_word=curr_strings[prev_model],
                symbol=string_to_compress[i],
                it=prev_model)

            bits_needed_for_curr_symbol = - math.log2(tmp_prob)

            self.list_of_bits_per_symbol.append(bits_needed_for_curr_symbol)
            self.number_of_bits += bits_needed_for_curr_symbol

        # print("Mixture: needed an average of %f bits per symbol" % (self.number_of_bits / len(string_to_compress)))
        return self.list_of_bits_per_symbol, self.number_of_bits

    def compress_string_based_on_models(self, string_to_compress):
        self._reset_number_of_bits()
        self.list_of_bits_per_symbol = []
        assert (dict() not in self.models_learned)

        # if self.with_neural_network:
        #     self._compute_optimal_compressions(string_to_compress)
        #     self._learn_neural_network(string_to_compress)

        log2_alphabet = math.log2(self.alphabet_size)
        min_k = (min(self.model_orders))
        max_k = (max(self.model_orders))

        # TODO - change this for optimal result
        for i in range(0, max_k):
            self.number_of_bits += float(log2_alphabet)
            self.list_of_bits_per_symbol.append(log2_alphabet)

        # initializing lists/arrays/whatever
        curr_strings = ["" for x in range(self.number_of_models)]
        curr_probs = [0.0 for x in range(self.number_of_models)]
        p_k_n = [1.0 for x in range(self.number_of_models)]

        # i is the position being compressed
        for i in range(max_k, len(string_to_compress)):

            for it in range(self.number_of_models):
                curr_strings[it] = string_to_compress[i - self.model_orders[it]:i]

            for it in range(self.number_of_models):
                curr_probs[it] = self.lidstone_estimate_probability_for_symbol(
                    curr_strings[it], string_to_compress[i], it)

            # calculate "recursive" function
            for it in range(self.number_of_models):
                p_k_n[it] = math.pow(self.model_weights[it], self.forgetting_factor) * curr_probs[it]

            # update model weights
            for it in range(self.number_of_models):
                # print("p_k_n[%d] = %f" % (it, p_k_n[it]))
                self.model_weights[it] = p_k_n[it] / float(sum(p_k_n))

            # print("sum(p_k_n) = %f \n\n" % float(sum(p_k_n)))
            # print("model weights = %s" % self.model_weights)

            # calculate bits needed
            bits_needed_for_curr_symbol = 0
            for it in range(self.number_of_models):
                bits_needed_for_curr_symbol += - (math.log2(curr_probs[it])) * self.model_weights[it]

            self.list_of_bits_per_symbol.append(bits_needed_for_curr_symbol)
            self.number_of_bits += bits_needed_for_curr_symbol

        # print("Mixture: needed an average of %f bits per symbol" % (self.number_of_bits / len(string_to_compress)))
        return self.list_of_bits_per_symbol, self.number_of_bits


def main():
    a = "ADABBDBABADBADBABDABBDABCABBDABDBBEABEBAEBBEBBABDDBACABDBABDABDBBADBADBBCBABCABBDBABEBEABBCBABDBDBABDBABEBEBABDBABDBDABDABABCABADBDABDBBADBBBDBCABDBADBABDBADABBDBABADBADBABDABBDABCABBDABDBBEABEBAEBBEBBABDDBACABDBABDABDBBADBADBBCBABCABBDBABEBEABBCBABDBDBABDBABEBEBABDBABDBDABDABABCABADBDABDBBADBBBDBCABDBADBABDBADABBDBABADBADBABDABBDABCABBDABDBBEABEBAEBBEBBABDDBACABDBABDABDBBADBADBBCBABCABBDBABEBEABBCBABDBDBABDBABEBEBABDBABDBDABDABABCABADBDABDBBADBBBDBCABDBADBABDBADABBDBABADBADBABDABBDABCABBDABDBBEABEBAEBBEBBABDDBACABDBABDABDBBADBADBBCBABCABBDBABEBEABBCBABDBDBABDBABEBEBABDBABDBDABDABABCABADBDABDBBADBBBDBCABDBADBABDB"
    b = "ABCABDBADBADBABDBADBABDBABDABCABBABDBDBBADBDABDBABCABBABABABABADBDABBBADBABCABBDBADBABDBABDABCABCABDBADBDBABDBABDBABCABBABABABABDABDBABCABDBABDABDBACABDBABDBABDABBDBACBADBABDABABCABDBADBADBABDBADBABDBABDABCABBABDBDBBADBDABDBABCABBABABABABADBDABBBADBABCABBDBADBABDBABDABCABCABDBADBDBABDBABDBABCABBABABABABDABDBABCABDBABDABDBACABDBABDBABDABBDBACBADBABDABABCABDBADBADBABDBADBABDBABDABCABBABDBDBBADBDABDBABCABBABABABABADBDABBBADBABCABBDBADBABDBABDABCABCABDBADBDBABDBABDBABCABBABABABABDABDBABCABDBABDABDBACABDBABDBABDABBDBACBADBABDABABCABDBADBADBABDBADBABDBABDABCABBABDBDBBADBDABDBABCABBABABABABADBDABBBADBABCABBDBADBABDBABDABCABCABDBADBDBABDBABDBABCABBABABABABDABDBABCABDBABDABDBACABDBABDBABDABBDBACBADBABDAB"
    alph_size = 6

    # path_test_1 = "Data/Psicologia_Nojo/SAX/N1.npy"
    # path_test_2 = "Data/Psicologia_Nojo/SAX/N2.npy"
    # path_test_16 = "Data/Psicologia_Nojo/SAX/N16.npy"
    # file_content1 = (np.load(path_test_1))
    # file_content2 = (np.load(path_test_2))
    # file_content16 = (np.load(path_test_16))
    # str_test1 = ''.join(file_content1).upper()
    # str_test2 = ''.join(file_content2).upper()
    # str_test16 = ''.join(file_content16).upper()

    # print("Len str1 = %d (%d heartbeats)" % (len(str_test1), len(str_test1) / 200))
    # print("Len str11 = %d (%d heartbeats)" % (len(str_test11), len(str_test11) / 200))

    fcm = FCMMixtures(alphab_size=alph_size, model_orders=[1, 3, 5], model_alphas=[1, 1, 1],
                      model_initial_weights=None,
                      forgetting_factor=0.99)

    # 29189,29729,29827
    # print("Learning models...")
    # fcm.learn_models_from_string(a)
    # time2 = time()
    # print("Took %.3f seconds to build model" % (time2 - time1))
    #
    # print("Compressing %s" % b)
    #
    # print("Took %.3f seconds to test" % (time() - time2))

    # fcm.print_models_learned()
    # fcm.print_details_of_models_learned()
    # fcm.print_memory_size_used()

    # print("%d bytes" % (bits / 8))

    # fcm._compute_model_chain(a)

    # TODO teste 1:
    fcm.learn_models_from_string(a)
    list_bits_chain, bits_chain = fcm.compress_string_based_on_models_and_model_chain(string_for_model_chain=b,
                                                                                      string_to_compress=b)

    list_bits_mixture, bits_mixture = fcm.compress_string_based_on_models(b)

    # # TODO teste 2:
    # a = str_test1
    # b = str_test16
    # fcm.learn_models_from_string(a)
    # list_bits_chain, bits_chain = fcm.compress_string_based_on_models_and_model_chain(string_for_model_chain=b,
    #                                                                                   string_to_compress=b)
    #
    # list_bits_mixture, bits_mixture = fcm.compress_string_based_on_models(b)
    #

    # PRINT RESULTS
    print("Using mixtures: ", bits_mixture, np.average(list_bits_mixture))
    print("Using chains: ", bits_chain, np.average(list_bits_chain))


if __name__ == "__main__":
    main()
