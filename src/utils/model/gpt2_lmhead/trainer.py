from transformers import Trainer
import torch
from utils.utils import Utils
import numpy as np

class GPT2LMHeadTrainer(Trainer):
    def __init__(self, encoder_model=None, encoder_model_is_fixed=True, custom_weights=None, embeddings=None, embedding_offset_lookup_table=None, device_from_args=None, t5_tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_model = encoder_model
        self.encoder_model_is_fixed = encoder_model_is_fixed
        self.custom_weights = custom_weights
        self.__embeddings = embeddings
        self.__embedding_offset_lookup_table = embedding_offset_lookup_table
        self.__device = device_from_args
        self.__t5_tokenizer = t5_tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        #print(inputs.keys())
        input_sequence = inputs["go_input_ids"]
        if self.encoder_model:
            if self.encoder_model_is_fixed:
                with torch.no_grad():
                    last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
            else:
                last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
        else:
            
            #print("inputs prot_input_ids shape is ", inputs["prot_input_ids"].shape)
            last_hidden_state = np.zeros((*inputs["prot_input_ids"].shape, 1024), dtype=np.float32)
            #print("last hidden state has been formed with shape", last_hidden_state.shape)
            idx = 0
            #prot_sequences = self.__t5_tokenizer.batch_decode(inputs["prot_input_ids"], skip_special_tokens=True)
            prot_sequences = inputs["prot_sequence"]
            for prot_sequence in prot_sequences:
                #print("PROT INPUT IDS SHAPE IS")
                #print(prot_input_ids.shape)
                #print(last_hidden_state.shape)
                
                #print("decoded prot sequence is ", prot_sequence)
                prot_sequence_hash = Utils.hash_prot_seq(prot_sequence)

                offset = self.__embedding_offset_lookup_table[prot_sequence_hash]
                length = min(len(prot_sequence.replace(" ", "").strip()), inputs["prot_input_ids"][0].shape[0])
                last_hidden_state[idx, :length, :] = self.__embeddings[offset : offset + length]
                idx += 1
            last_hidden_state = torch.from_numpy(last_hidden_state).to(self.__device)
            #print(last_hidden_state.shape)
            #print("last hidden state dtype", last_hidden_state.dtype)
            #print("input sequence dtype", input_sequence.dtype)

        outputs = model(input_ids=input_sequence, encoder_hidden_states=last_hidden_state, labels=input_sequence)
        if self.custom_weights is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss
        else:
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.custom_weights)
            loss = loss_fct(logits.view(-1, len(self.custom_weights)), input_sequence.view(-1))
            return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model_inputs = {
            "go_input_ids": inputs["go_input_ids"],
            "prot_input_ids": inputs["prot_input_ids"],
            "prot_attention_mask": inputs["prot_attention_mask"],
            "labels": inputs["go_input_ids"] if "labels" not in inputs else inputs["labels"],
            "prot_sequence": inputs["prot_sequence"]
        }
        
        return super().prediction_step(
            model,
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )