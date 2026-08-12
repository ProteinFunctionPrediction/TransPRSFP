import torch

class DatasetUtils:
    @staticmethod
    def generate_torch_dataset_compatible_dataset_iterator(dataset, source_tokenizer, target_tokenizer, batch_size, maxlen, embedding_offset_lookup_table=None):
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
                prot_sequence = batch[i][0]
                ret["prot_sequence"] = prot_sequence
                ret["prot_input_ids"] = tokenized_sequences[i]
                
                go_input_ids = tokenized_go_terms.tolist()[i]
                prot_attention_mask = tokenized_sequences_attention_mask[i]

                ret["prot_attention_mask"] = prot_attention_mask
                ret["go_input_ids"] = go_input_ids
                ret["labels"] = DatasetUtils.construct_residue_labels(go_input_ids=go_input_ids, prot_attention_mask=prot_attention_mask)
                if embedding_offset_lookup_table is not None:
                    prot_sequence_hash = Utils.hash_prot_seq(prot_sequence)
                    offset, length = embedding_offset_lookup_table[prot_sequence_hash]
                    ret["embedding_offset"] = offset
                    # +1: EOS
                    ret["embedding_length"] = length + 1
                yield ret

    @staticmethod
    def construct_residue_labels(go_input_ids, prot_attention_mask, ignore_index=-100):
        batch_size, sequence_length = prot_attention_mask.shape
        labels = torch.full(
            (batch_size, sequence_length),
            fill_value=ignore_index,
            dtype=torch.long,
            device=go_input_ids.device,
        )

        # remove SOS (GO side)
        shifted_labels = go_input_ids[:, 1:]
        copy_length = min(sequence_length, shifted_labels.shape[1])

        labels[:, :copy_length] = shifted_labels[:, :copy_length]

        # -1 -> remove EOS
        residue_lengths = prot_attention_mask.sum(dim=1).long() - 1

        #residue_lengths = residue_lengths.clamp(min=0, max=sequence_length)
        positions = torch.arange(sequence_length, device=go_input_ids.device).unsqueeze(0)

        residue_attention_mask = positions < residue_lengths.unsquueze(1)

        # EOS is also removed at the GO side now
        labels = labels.masked_fill(~residue_attention_mask, ignore_index)

        return labels