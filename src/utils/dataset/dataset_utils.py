import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from universal.settings.settings import Settings
import numpy as np

class DatasetUtils:
    
    @staticmethod
    def load(fn):
        with open(fn, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def infer_maximum_length(dataset):
        max_length = 0
        for protein_sequence, go_terms in dataset:
            length = len(protein_sequence.replace(' ', ''))
            if length > max_length:
                max_length = length
        
        return max_length
    
    
    @staticmethod
    def get_training_count(training_dataset_ratio: float, n: int) -> int:
        # n: number of samples in the dataset
        n_train = round(n * training_dataset_ratio)
        if n_train == 0:
            n_train += 1
        
        n_validation = n - n_train
        
        if n_validation <= 0:
            raise RuntimeError("The number of samples in the dataset is inadequate!")
        
        assert n_train + n_validation == n
        
        return n_train

    @staticmethod
    def get_go_term_count(dataset) -> int:
        tf_tokenizer = Tokenizer(oov_token='<OOV>', filters='')
        tf_tokenizer.fit_on_texts(dataset[:, 1])
        
        # - 3: <sos>, <eos>, <OOV>
        return len(tf_tokenizer.word_index) - 3
    
    @staticmethod
    def generate_batch_iterator(dataset, source_tokenizer, target_tokenizer, batch_size):
        """
        Takes the dataset to be used and returns a generator that yields batches 
        consisting of tokenized [input, output] elements. Yielded lists are of the form
        [[[1, 5, 2, 8, ...], [8, 9, 15, 20, 35, ...]], ...] and has <batch_size> elements.
        dataset -- Output of the prepare_dataset function
        source_tokenizer -- a T5Tokenizer pretrained on a protein sequence dataset
        target_tokenizer -- a tf.keras.preprocessing.text.Tokenizer that is fit to the output GO terms
        """
        
        idx = 0
        while idx < len(dataset):
            ret = []
            batch = dataset[idx:idx+batch_size]
            
            
            batch_protein_sequences = batch[:, 0]
            tokenized_sequences = source_tokenizer.batch_encode_plus(\
                                                                    batch_protein_sequences,\
                                                                    add_special_tokens=True,\
                                                                    padding="longest").input_ids

            batch_go_term_sequences = batch[:, 1]
            if target_tokenizer is not None:
                tokenized_go_terms = pad_sequences(target_tokenizer.texts_to_sequences(batch_go_term_sequences), padding='post')
            else:
                tokenized_go_terms = []
                for i in range(batch.shape[0]):
                    tokenized_go_terms.append([Settings.TRANSFORMER_TRG_PAD_IDX])

            for i in range(batch.shape[0]):
                ret.append([tokenized_sequences[i], tokenized_go_terms[i]])
            
            idx += batch_size
            
            yield ret

    @staticmethod
    def split_batch_into_X_y(batch):
        """
        Splits the given batch into two separate numpy arrays: X and y, 
        where X is the list of input features and y is the list of output labels.
        batch -- each one of batches yielded by the batch iterator returned by generate_batch_iterator function
        """
        
        X = []
        y = []
        
        for x_elem, y_elem in batch:
            X.append(x_elem)
            y.append(y_elem)
        
        return np.asarray(X), np.asarray(y)

    @staticmethod
    def count_go_term_datapoints(dataset):
        result = {}
        for prot_seq, go_terms in dataset:
            s = set(go_terms.split()[1:-1])
            for go_term in s:
                if go_term not in result:
                    result[go_term] = 0
                result[go_term] += 1
        return result

    @staticmethod
    def split_train_val(dataset, val_count):
        val = []
        counts = DatasetUtils.count_go_term_datapoints(dataset)
        remove_proteins = set()
        count = 0
        for i in dataset:
            prot_seq, go_terms = i
            s = set(go_terms.split()[1:-1])
            skip = False
            for go_term in s:
                if counts[go_term] <= 1:
                    skip = True
                    break
            if skip:
                continue

            for go_term in s:
                counts[go_term] -= 1
            val.append(i)
            remove_proteins.add(prot_seq)
            count += 1

            if count == val_count:
                break

        train = []
        for i in dataset:
            prot_seq, go_terms = i
            if prot_seq not in remove_proteins:
                train.append(i)

        return np.asarray(train), np.asarray(val)
