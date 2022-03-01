import abc


class ABCFCMCompressor(metaclass=abc.ABCMeta):
    """Abstract Base Class Definition"""

    @abc.abstractmethod
    def learn_models_from_string(self, np_string):
        pass

    @abc.abstractmethod
    def print_models_learned(self):
        pass

    @abc.abstractmethod
    def print_memory_size_used(self):
        pass

    @abc.abstractmethod
    def compress_string_based_on_models(self, string_to_compress):
        pass




