import torch
import numpy as np
import gc
import torch

class Utils:
    
    @staticmethod
    def free_gpu_memory():
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_fp_tp_fn_tn(y_true, y_pred, total_go_term_count, trg_pad_idx = -1, empty_token=-1):
        fp, tp, fn = 0, 0, 0
                
        if type(y_true) == torch.Tensor:
            y_true = list(y_true.cpu().numpy())
        
        if type(y_pred) == torch.Tensor:
            y_pred = list(y_pred.cpu().numpy())
                
        y_true = list(np.unique(np.asarray(y_true)))
        y_pred = list(np.unique(np.asarray(y_pred)))
        
        
        if trg_pad_idx in y_true:
            y_true.remove(trg_pad_idx)
        if trg_pad_idx in y_pred:
            y_pred.remove(trg_pad_idx)
            
        if empty_token in y_true:
            y_true.remove(empty_token)
        if empty_token in y_pred:
            y_pred.remove(empty_token)
        
        for token in y_pred:
            if token not in y_true:
                fp += 1
            else:
                tp += 1
        
        for token in y_true:
            if token not in y_pred:
                fn += 1
        
        tn = total_go_term_count - (fp + tp + fn)

        return fp, tp, fn, tn
    
    @staticmethod
    def get_fp_tp_fn_tn_token_based(y_true, y_pred, total_go_term_count, trg_pad_idx = 0, empty_token=-1):
        fp, tp, fn, tn = 0, 0, 0, 0
            
        if type(y_true) == torch.Tensor:
            y_true = list(y_true.cpu().numpy())
        
        if type(y_pred) == torch.Tensor:
            y_pred = list(y_pred.cpu().numpy())
        
        try:
            assert len(y_true) == len(y_pred)
        except AssertionError as e:
            print(len(y_true), len(y_pred))
            assert len(y_true) == len(y_pred) # let the execution terminate
        
        for i in range(len(y_true)):
            true_token = y_true[i]
            pred_token = y_pred[i]
            if true_token == empty_token and pred_token == empty_token:
                tn += 1
            elif true_token == empty_token and pred_token != empty_token:
                fp += 1 # fp for predicting an incorrect token
            elif true_token != empty_token and pred_token != empty_token:
                if true_token == pred_token:
                    tp += 1
                else:
                    fn += 1 # fn for failing to predict the correct token
                    fp += 1 # fp for predicting an incorrect token
            elif true_token != empty_token and pred_token == empty_token:
                fn += 1 # fn for failing to predict the correct token
        
        return fp, tp, fn, tn
        
        
    @staticmethod    
    def precision(tp, fp):
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    @staticmethod
    def recall(tp, fn):
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)

    @staticmethod
    def accuracy(fp, tp, fn, tn):
        return (tp + tn) / (fp + tp + fn + tn)

    @staticmethod
    def f1(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * ((precision * recall) / (precision + recall))
    
    @staticmethod
    def softmax(z):
        return np.exp(z)/np.sum(np.exp(z), axis=0)
    
    @staticmethod
    def build_reverse_index(go_term_to_index):
        reverse_index = dict()
        for key, value in go_term_to_index.items():
            assert value not in reverse_index
            reverse_index[value] = key
        
        return reverse_index

    @staticmethod
    def get_unique_tokens(l):
        return list(set(l))

    @staticmethod
    def convert_tokens_to_str_go_terms(reverse_go_term_to_index, predictions):
        predictions = Utils.get_unique_tokens(predictions)
        
        return [reverse_go_term_to_index[i].upper() if reverse_go_term_to_index[i].lower().startswith('go') else reverse_go_term_to_index[i] for i in predictions]
    
    @staticmethod
    def match_ratio(y_true, y_pred):
        assert len(y_true) == len(y_pred)
        equal_count = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]: equal_count += 1
        return equal_count/len(y_true)

    @staticmethod
    def match_ratio_nonempty(y_true, y_pred, empty_token):
        assert len(y_true) == len(y_pred)
        equal_count, nonempty_count = 0, 0
        for i in range(len(y_true)):
            if y_true[i] != empty_token and y_true[i] == y_pred[i]:
                equal_count += 1
            if y_true[i] != empty_token:
                nonempty_count += 1

        return equal_count/nonempty_count
    
    @staticmethod
    def generate_lookup_matrix(counts_of_types):
        """
        This function generates and returns a lookup matrix which can be used to map a number to a list
        [index_of_element_in_type_1, index_of_element_in_type_2, ...]. This lookup matrix then can be
        used to get all possible combinations of different elements found in each type.
        
        For example, if we have 2 different hyperparameters, namely, learning rate and number of heads,
        and 2 different values for learning rate and 3 different values for number of heads,
        the counts_of_types list would be [2, 3]. With such an input, this function would generate the
        following matrix:
        [[0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2]]
        """
        
        product_of_dims = 1
        for i in counts_of_types:
            product_of_dims *= i
        
        ret = np.zeros((product_of_dims, len(counts_of_types)))
        for i in range(len(counts_of_types)):
            if i == len(counts_of_types) - 1:
                digit_repetition_times = 1
            else:
                right_part = counts_of_types[i + 1:]
                digit_repetition_times = 1
                for number in right_part:
                    digit_repetition_times *= number
            block_repetition_times = int(product_of_dims/(digit_repetition_times * counts_of_types[i]))
            for j in range(block_repetition_times):
                for k in range(counts_of_types[i]):
                    for p in range(digit_repetition_times):
                        ret[int(j * counts_of_types[i] * digit_repetition_times + k * digit_repetition_times + p), i] = k
        return ret.astype(int), product_of_dims

    def create_model_name_from_hyperparameters(hyperparameters):
        result = ""
        for key, value in hyperparameters.items():
            result += key + "_" + str(value) + "."
        return result[:-1]

