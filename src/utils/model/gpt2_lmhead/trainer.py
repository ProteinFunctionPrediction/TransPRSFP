from typing import Optional

from transformers import Trainer
import torch
from utils.utils import Utils
import numpy as np
from transformers.trainer_pt_utils import LengthGroupedSampler

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

    def _get_train_sampler(self):
        if self.args.group_by_length:
            lengths = [int(torch.as_tensor(x["prot_attention_mask"]).sum().item()) for x in self.train_dataset.data]
            return LengthGroupedSampler(batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps, lengths=lengths)

        return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False):
        input_sequence = inputs["go_input_ids"]
        if self.encoder_model:
            if self.encoder_model_is_fixed:
                with torch.no_grad():
                    last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
            else:
                last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
        else:
            last_hidden_state = inputs["last_hidden_states"]

        outputs = model(input_ids=input_sequence, encoder_hidden_states=last_hidden_state, encoder_attention_mask=inputs["prot_attention_mask"], labels=input_sequence)
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
        }
        
        if "last_hidden_states" in inputs:
            model_inputs["last_hidden_states"] = inputs["last_hidden_states"]

        return super().prediction_step(
            model,
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )