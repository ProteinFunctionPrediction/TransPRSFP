from transformers import Trainer
import torch

class GPT2LMHeadTrainer(Trainer):
    def __init__(self, encoder_model=None, custom_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_model=encoder_model
        self.custom_weights = custom_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        input_sequence = inputs["go_input_ids"]
        with torch.no_grad():
            last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
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
            "labels": inputs["go_input_ids"] if "labels" not in inputs else inputs["labels"]
        }
        
        return super().prediction_step(
            model,
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )