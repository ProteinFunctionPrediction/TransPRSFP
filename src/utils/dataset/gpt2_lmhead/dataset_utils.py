from universal.settings.settings import Settings
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DatasetUtils:
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
            ret = {}
            batch = dataset[idx:idx+batch_size]
            
            
            batch_protein_sequences = batch[:, 0]
            batch_encode_plus_output = source_tokenizer.batch_encode_plus(\
                                                                    batch_protein_sequences,\
                                                                    add_special_tokens=True,\
                                                                    padding="longest")

            tokenized_sequences = batch_encode_plus_output["input_ids"]
            tokenized_sequences_attention_mask = batch_encode_plus_output["attention_mask"]

            batch_go_term_sequences = batch[:, 1]
            if target_tokenizer is not None:
                tokenized_go_terms = pad_sequences(target_tokenizer.texts_to_sequences(batch_go_term_sequences), padding='post')
            else:
                tokenized_go_terms = []
                for i in range(batch.shape[0]):
                    tokenized_go_terms.append([Settings.TRANSFORMER_TRG_PAD_IDX])

            ret["prot_input_ids"] = tokenized_sequences
            ret["prot_attention_mask"] = tokenized_sequences_attention_mask
            ret["go_input_ids"] = tokenized_go_terms

            #for i in range(batch.shape[0]):
            #    ret.append([tokenized_sequences[i], tokenized_go_terms[i]])
            
            idx += batch_size
            
            yield ret
    
    @staticmethod
    def generate_torch_dataset_compatible_dataset_iterator(dataset, source_tokenizer, target_tokenizer, batch_size, maxlen):
        idx = 0
        while idx < len(dataset):
            batch = dataset[idx:idx+batch_size]
            
            
            batch_protein_sequences = batch[:, 0]
            batch_encode_plus_output = source_tokenizer.batch_encode_plus(\
                                                                    batch_protein_sequences,\
                                                                    add_special_tokens=True,\
                                                                    padding="max_length",
                                                                    max_length=maxlen,
                                                                    truncation=True)

            tokenized_sequences = batch_encode_plus_output["input_ids"]
            tokenized_sequences_attention_mask = batch_encode_plus_output["attention_mask"]
            
            #print(tokenized_sequences[0])
            #print(tokenized_sequences_attention_mask[0])

            batch_go_term_sequences = batch[:, 1]
            if target_tokenizer is not None:
                tokenized_go_terms = pad_sequences(target_tokenizer.texts_to_sequences(batch_go_term_sequences), padding='post', maxlen=maxlen, truncating='post')
            else:
                tokenized_go_terms = []
                for i in range(batch.shape[0]):
                    tokenized_go_terms.append([Settings.TRANSFORMER_TRG_PAD_IDX])

            idx += batch_size

            for i in range(len(tokenized_sequences)):
                ret = {}
                ret["prot_input_ids"] = tokenized_sequences[i]
                ret["prot_attention_mask"] = tokenized_sequences_attention_mask[i]
                ret["go_input_ids"] = tokenized_go_terms.tolist()[i]
                yield ret

            #for i in range(batch.shape[0]):
            #    ret.append([tokenized_sequences[i], tokenized_go_terms[i]])
            
            
