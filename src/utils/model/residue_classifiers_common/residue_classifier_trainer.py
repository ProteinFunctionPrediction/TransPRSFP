from abc import ABC
import torch
from transformers import Trainer

from transformers.trainer_pt_utils import LengthGroupedSampler

class ResidueClassifierTrainer(Trainer, ABC):
    def __init__(self, encoder_model=None, encoder_model_is_fixed=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder_model = encoder_model
        self.encoder_model_is_fixed = encoder_model_is_fixed

    def _get_train_sampler(self):
        if self.args.group_by_length:
            lengths = [int(torch.as_tensor(x["prot_attention_mask"]).sum().item()) for x in self.train_dataset.data]
            return LengthGroupedSampler(batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps, lengths=lengths)

        return super()._get_train_sampler()

    def create_scheduler(self, num_training_steps, optimizer=None):
        scheduler = super().create_scheduler(num_training_steps, optimizer=optimizer)
        if not hasattr(scheduler, "original_num_training_steps"):
            scheduler.original_num_training_steps = num_training_steps
        
        return scheduler
