import torch
import numpy as np
from model.classification_head.model import MultiLabelProteinClassifier
from universal.access.universal_access import UniversalAccess
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
from utils.model.model_utils import ModelUtils

class ClassificationHeadUtils(ModelUtils):
    def __init__(self, device) -> None:
        super().__init__(device)
    
    def classification_head_predict(self, sequence, model, tokenizer):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors="pt", padding=True).to(self.device)
            return self.classification_head_predict_by_tokens(inputs["input_ids"], inputs["attention_mask"], model)

    def classification_head_predict_by_tokens(self, input_ids, attention_mask, model):
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).to(self.device)
            probabilities = torch.sigmoid(logits)
            
            return probabilities.cpu().numpy()
    
    def classification_head_predict_by_sequence(self, sequence, model, tokenizer):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors="pt", padding=True).to(self.device)
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            probabilities = torch.sigmoid(logits)
            return probabilities.cpu().numpy()
    
    def one_hot_decode(self, one_hot_encoded):
        ret = []
        for i, item in enumerate(one_hot_encoded):
            if item != 0:
                ret.append(i)
        
        return ret

    def classification_head_predict_onezero_vector(self, input_ids, attention_mask, model, threshold=0.5):
        probs = self.classification_head_predict_by_tokens(input_ids, attention_mask, model)[0]
        ret = []
        for i in probs:
            if i >= threshold:
                ret.append(1)
            else:
                ret.append(0)
        return ret

    def classification_head_predict_token_vector(self, input_ids, attention_mask, model, threshold=0.5):
        onezero_vector = self.classification_head_predict_onezero_vector(input_ids, attention_mask, model, threshold)
        return self.one_hot_decode(onezero_vector)
    
    def run_classification_head_prediction(self, dataloader, model, threshold=0.5, batch_size=16, caller=None):  
    
        classification_predictions = []
        
        for i, (inputs, labels) in enumerate(iter(dataloader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            for j, input_id_tensor in enumerate(input_ids):
                attention_mask_tensor = attention_mask[j]
                count = i * batch_size + j + 1
                predicted_tokens = self.classification_head_predict_token_vector(input_id_tensor, attention_mask_tensor, model, threshold)
                classification_predictions.append(predicted_tokens)
                
                if caller is not None:
                    caller.notify(count - 1, " ".join([model.get_config().reverse_go_term_to_index[_token_idx] for _token_idx in predicted_tokens]))
            

        return classification_predictions

    def train(self, num_epochs: int, lr: float, model: MultiLabelProteinClassifier, go_term_count: int,
              train_loader: DataLoader, val_loader: DataLoader,
              save_per_epoch: int, model_save_dir: str) -> None:

        

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                labels = labels.to(self.device)

                logits = model(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            UniversalAccess.output.write(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}")

            # Validation
            model.eval()
            val_loss = 0.0
            for inputs, labels in val_loader:
                with torch.no_grad():
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)
                    labels = labels.to(self.device)

                    logits = model(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
                    loss = criterion(logits, labels)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            UniversalAccess.output.write(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")
            
            if save_per_epoch > 0 and (epoch + 1) % save_per_epoch == 0:
                UniversalAccess.output.write(f"End of epoch {epoch + 1}: saving model...")
                model.save(os.path.join(model_save_dir, "epoch_" + str(epoch + 1)))
                UniversalAccess.output.write(f"Done!")
        model.save(os.path.join(model_save_dir, "end_of_training"))
