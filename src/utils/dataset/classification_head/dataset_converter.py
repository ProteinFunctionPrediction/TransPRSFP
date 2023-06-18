from tensorflow.keras.preprocessing.text import Tokenizer
import copy

class DatasetConverter:
    def __init__(self, old_format_dataset, ignore_tokens):
        self.old_format_dataset = old_format_dataset
        self.ignore_tokens = ignore_tokens
        self.tf_tokenizer = Tokenizer(oov_token='<OOV>', filters='')
        self.tf_tokenizer.fit_on_texts(old_format_dataset[:, 1])
        self.go_term_count = len(self.tf_tokenizer.word_index)
        for token in self.ignore_tokens:
            if token in self.tf_tokenizer.word_index:
                self.go_term_count -= 1
        
        self.go_term_to_index_map = {}
        self.all_go_terms = [] # i.e., index_to_go_term_map
        
        self._build_go_term_to_index_map()
        assert len(self.all_go_terms) == self.go_term_count
        
        self.data = []
        
        
    def _build_go_term_to_index_map(self):
        tf_tokenizer_word_index = copy.deepcopy(self.tf_tokenizer.word_index)
        for token in self.ignore_tokens:
            if token in tf_tokenizer_word_index:
                del tf_tokenizer_word_index[token]
        
        self.all_go_terms = [i.upper() for i in copy.deepcopy(list(tf_tokenizer_word_index.keys()))]
        self.all_go_terms.sort()
        
        for i, go_term in enumerate(self.all_go_terms):
            self.go_term_to_index_map[go_term] = i
    
    def convert(self):
        for prot_sequence, go_term_sequence in self.old_format_dataset:
            for token in self.ignore_tokens:
                go_term_sequence = go_term_sequence.replace(token, "")
            go_term_sequence = go_term_sequence.strip()
            go_term_sequence_split = [i for i in go_term_sequence.split() if len(i.strip()) > 0]
            
            labels = [0 for i in range(self.go_term_count)]
            
            for go_term in go_term_sequence_split:
                index = self.go_term_to_index_map[go_term]
                labels[index] = 1
            
            self.data.append((prot_sequence, labels))

    def get_new_dataset(self):
        return self.data