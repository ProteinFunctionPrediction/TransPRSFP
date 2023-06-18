import torch.nn as nn
from model.model import Model

class MultiLabelProteinClassifier(nn.Module, Model):
    def __init__(self, prot_t5_model, num_labels):
        super(MultiLabelProteinClassifier, self).__init__()
        self.encoder = prot_t5_model
        self.classifier = nn.Linear(1024, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(sequence_output)
        return logits

    def save(self, model_save_dir: str) -> None:
        Model.save(self, model_save_dir)
