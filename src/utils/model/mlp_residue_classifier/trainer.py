import torch
from transformers import Trainer
from utils.dataset.mlp_residue_classifier.dataset_utils import DatasetUtils as MLPResidueClassifierDatasetUtils

class MLPResidueClassifierTrainer(Trainer):
    def __init__(self, encoder_model=None, encoder_model_is_fixed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder_model = encoder_model
        self.encoder_model_is_fixed = encoder_model_is_fixed
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.encoder_model:
            if self.encoder_model_is_fixed:
                with torch.no_grad():
                    last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
            else:
                last_hidden_state = self.encoder_model(input_ids=inputs["prot_input_ids"], attention_mask=inputs["prot_attention_mask"]).last_hidden_state
        else:
            last_hidden_state = inputs["last_hidden_states"]
                
        outputs = model(last_hidden_state=last_hidden_state, labels=inputs["labels"])

        return (outputs.loss, outputs) if return_outputs else outputs.loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model_inputs = {
            "go_input_ids": inputs["go_input_ids"],
            "prot_input_ids": inputs["prot_input_ids"],
            "prot_attention_mask": inputs["prot_attention_mask"],
            "labels": inputs["labels"]
        }

        if "last_hidden_states" in inputs:
            model_inputs["last_hidden_states"] = inputs["last_hidden_states"]
        
        return super().prediction_step(
            model,
            model_inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys
        )