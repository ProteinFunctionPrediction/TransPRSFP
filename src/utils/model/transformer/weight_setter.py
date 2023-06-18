class WeightSetter:
    def __init__(self, dataset, tf_tokenizer, C, pad_index):
        self.weights: list = []
        self.token_count_map: dict = {}
        
        
        self.dataset = dataset
        self.tf_tokenizer = tf_tokenizer
        self.C = C
        self.pad_index = pad_index
        
        
        self._build_token_count_map()
        self._set_weights()
    
    def _build_token_count_map(self):
        for i in self.dataset:
            target_sequence = i[1].strip().split()
            for token in target_sequence:
                token = token.upper()
                if token in self.token_count_map:
                    self.token_count_map[token] += 1
                else:
                    self.token_count_map[token] = 1
    
    def _set_weights(self):
        max_occurrence = max(list(self.token_count_map.values()))
        self.weights = [0 for i in range(self.C)] # weights list must be of length C
        
        for token, index in self.tf_tokenizer.word_index.items():
            token = token.upper()
            if token not in self.token_count_map:
                self.weights[index] = len(self.dataset) / self.C
            else:
                self.weights[index] = (1 / self.token_count_map[token]) * len(self.dataset) / self.C
        
        self.weights[self.pad_index] = 0
        
    def get_weights(self):
        return self.weights