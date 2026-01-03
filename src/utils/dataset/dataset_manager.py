import numpy as np

from utils.dataset.dataset_utils import DatasetUtils
from universal.settings.settings import Settings
from tensorflow.keras.preprocessing.text import Tokenizer

class DatasetManager:
    def __init__(self, dataset):
        self.__dataset = dataset
        # The unwrapped version of the augmented dataset 
        # (Applicaple if and only if the dataset is augmented. Otherwise, it would simply be the same as the dataset itself)
        # The word "augmented" in the source code is used as a shortcut to mean that the dataset is a numpy array of such instances:
        # [protein_sequence, [go_term_sequences_str1, go_term_sequences_str2, ...]]
        # The unwrapping here means to convert the format above to the normal dataset format
        self.__dataset_unwrapped = dataset
        self.__is_dataset_augmented = DatasetUtils.is_dataset_augmented(dataset)
        if self.__is_dataset_augmented:
            self.__dataset_unwrapped = DatasetUtils.unwrap_augmented_dataset(dataset)
        
        self.__tf_tokenizer = self.__prepare_tf_tokenizer()
        self.__go_term_count = self.__compute_go_term_count()
                
    def __prepare_tf_tokenizer(self):
        tf_tokenizer = Tokenizer(oov_token=Settings.TRANSFORMER_OOV_TOKEN, filters='')
        tf_tokenizer.fit_on_texts(self.__dataset_unwrapped[:, 1])
        return tf_tokenizer
    
    def __compute_go_term_count(self):
        return len(self.__tf_tokenizer.word_index) - 3
    
    def get_training_count(self, training_dataset_ratio):
        return DatasetUtils.get_training_count(training_dataset_ratio, len(self.__dataset_unwrapped))

    def get_datapoint_count(self):
        return len(self.__dataset_unwrapped)

    def get_go_term_count(self):
        return self.__go_term_count
    
    def get_tf_tokenizer(self):
        return self.__tf_tokenizer
    
    def get_dataset(self):
        return self.__dataset
    
    def get_unwrapped_dataset(self):
        return self.__dataset_unwrapped

    def is_dataset_augmented(self):
        return self.__is_dataset_augmented
    
    def shuffle(self):
        np.random.shuffle(self.__dataset)
        self.__dataset_unwrapped = self.__dataset
        if self.__is_dataset_augmented:
            self.__dataset_unwrapped = DatasetUtils.unwrap_augmented_dataset(self.__dataset)