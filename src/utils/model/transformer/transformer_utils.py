from model.transformer.model import Transformer
from utils.model.model_utils import ModelUtils
import torch
import torch.nn as nn
from utils.dataset.dataset_utils import DatasetUtils
from universal.settings.settings import Settings
from io import StringIO
import json
import numpy as np
from utils.utils import Utils
import os
import gc
from torch.utils.tensorboard import SummaryWriter
from model.transformer.config.transformer_model_config import TransformerModelConfig
from model.model_navigator import ModelNavigator


class TransformerUtils(ModelUtils):
    def __init__(self, device) -> None:
        super().__init__(device)
    
    def run_transformer_prediction(self,
                                   batches,
                                   model: Transformer,
                                   pred_SOS_token_id,
                                   pred_EOS_token_id,
                                   pred_EMPTY_token_id,
                                   pred_reverse_word_index,
                                   true_SOS_token_id,
                                   true_EOS_token_id,
                                   true_EMPTY_token_id,
                                   true_reverse_word_index,
                                   batch_size=8,
                                   caller=None,
                                   prefix="",
                                   compute_go_based_metrics=False,
                                   save_go_based_metrics=True,
                                   go_based_metrics_filepath="",
                                   compute_metrics=False):
    
        if compute_metrics:
            assert true_SOS_token_id is not None
            assert true_EOS_token_id is not None
            assert true_EMPTY_token_id is not None
    
        total_go_term_count = len(pred_reverse_word_index) - 1
        dataloader = batches
        predictions = []
        
        if compute_go_based_metrics:
            go_based_metrics = Utils.build_go_based_metrics_map_from_reverse_go_term_indices(pred_reverse_word_index,
                                                                                             true_reverse_word_index)
        
        
        total_sample_count = 0
        running_metrics = {prefix + "precision": 0.0,
                           prefix + "recall": 0.0,
                           prefix + "accuracy": 0.0,
                           prefix + "f1": 0.0,
                           prefix + "match_ratio": 0.0,
                           prefix + "match_ratio_nonempty": 0.0,
                           prefix + "token_based_precision": 0.0,
                           prefix + "token_based_accuracy": 0.0,
                           prefix + "token_based_recall": 0.0,
                           prefix + "token_based_f1": 0.0}
        
        with torch.no_grad():
            for batch_idx in range(len(dataloader)):
                batch = dataloader[batch_idx]
                X, y = DatasetUtils.split_batch_into_X_y(batch)
                X, y = torch.tensor(X, dtype=torch.long, device=self.device), torch.tensor(y, dtype=torch.long, device=self.device)
                for idx in range(len(X)):
                    count = batch_idx * batch_size + idx + 1
                    
                    sample_X = X[idx].unsqueeze(0)
                    sample_y = y[idx][y[idx] != Settings.TRANSFORMER_TRG_PAD_IDX][1:-1]
                        
                    prediction = self.predict(model,
                                              sample_X,
                                              SOS_token=pred_SOS_token_id,
                                              EOS_token=pred_EOS_token_id,
                                              max_length=len(sample_X[0][sample_X[0] != 0]))[1:-1]
                        
                    predictions.append(prediction)

                    if len(sample_y) > 0 and sample_y[0] == true_SOS_token_id:
                        sample_y = sample_y[1:]
                
                    if len(sample_y) > 0 and sample_y[-1] == true_EOS_token_id:
                        sample_y = sample_y[:-1]
                        
                    
                    if compute_metrics:
                        assert len(prediction) == len(sample_y)
                        unique_prediction = list(np.unique(np.asarray(prediction)))
                        
                        sample_y_str_list = Utils.convert_idx_list_to_strings(sample_y, true_reverse_word_index)
                        unique_prediction_str_list = Utils.convert_idx_list_to_strings(unique_prediction, pred_reverse_word_index)
                        prediction_str_list = Utils.convert_idx_list_to_strings(prediction[:len(sample_y)], pred_reverse_word_index)

                        fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(sample_y_str_list,
                                                               unique_prediction_str_list,
                                                               total_go_term_count,
                                                               true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                               pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                        precision_score, recall_score, accuracy_score = Utils.precision(tp, fp), Utils.recall(tp, fn), Utils.accuracy(fp, tp, fn, tn)
                        f1_score = Utils.f1(precision_score, recall_score)
                    
                        match_ratio_score = Utils.match_ratio(sample_y_str_list, prediction_str_list)
                        match_ratio_nonempty_score = Utils.match_ratio_nonempty(sample_y_str_list, prediction_str_list, true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())

                        # _t: abbreviation for _token
                        fp_t, tp_t, fn_t, tn_t = Utils.get_fp_tp_fn_tn_token_based(sample_y_str_list,
                                                                                   prediction_str_list,
                                                                                   total_go_term_count,
                                                                                   true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                                                   pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                        precision_score_t, recall_score_t, accuracy_score_t = Utils.precision(tp_t, fp_t), Utils.recall(tp_t, fn_t), \
                                                                            Utils.accuracy(fp_t, tp_t, fn_t, tn_t)
                        f1_score_t = Utils.f1(precision_score_t, recall_score_t)
                    
                        if compute_go_based_metrics:
                            Utils.update_go_based_metrics_map(sample_y_str_list,
                                                              prediction_str_list,
                                                              go_based_metrics)
                    
                        running_metrics["precision"] += precision_score
                        running_metrics["recall"] += recall_score
                        running_metrics["accuracy"] += accuracy_score
                        running_metrics["f1"] += f1_score
                    

                        running_metrics["match_ratio"] += match_ratio_score
                        running_metrics["match_ratio_nonempty"] += match_ratio_nonempty_score
                        running_metrics["token_based_precision"] += precision_score_t
                        running_metrics["token_based_recall"] += recall_score_t
                        running_metrics["token_based_accuracy"] += accuracy_score_t
                        running_metrics["token_based_f1"] += f1_score_t
                        total_sample_count += 1
                        
                        print({"count": total_sample_count,
                               "precision": precision_score,
                               "recall": recall_score,
                               "accuracy": accuracy_score,
                               "f1": f1_score,
                               "match_ratio": match_ratio_score,
                               "match_ratio_nonempty": match_ratio_nonempty_score,
                               "token_based_precision": precision_score_t,
                               "token_based_recall": recall_score_t,
                               "token_based_accuracy": accuracy_score_t,
                               "token_based_f1": f1_score_t})
                    
                    if caller is not None:
                        caller.notify(count - 1, self.post_process_prediction_as_str(self.post_process_prediction(model, prediction, pred_EMPTY_token_id)))
                    
            average_metrics = {}
            if compute_metrics:
                for key, value in running_metrics.items():
                    average_metrics[key] = value/total_sample_count

            if compute_go_based_metrics and save_go_based_metrics and go_based_metrics_filepath != "":
                if os.path.exists(go_based_metrics_filepath):
                    print(f"{go_based_metrics_filepath} already exists! Won't save GO term-based metrics:")
                    print(go_based_metrics)
                else:
                    try:
                        with open(go_based_metrics_filepath, "wb") as go_based_metrics_file:
                            pickle.dump(go_based_metrics, go_based_metrics_file)
                    except Exception as e:
                        print(f"An error occurred while saving GO term-based metrics to {os.path.abspath(go_based_metrics_filepath)}: {str(e)}")
                    else:
                        print(f"GO term-based metrics have been saved to {os.path.abspath(go_based_metrics_filepath)}")
            elif compute_go_based_metrics:
                print("GO term-based metrics:")
                print(go_based_metrics)

        return average_metrics, predictions

    def predict(self, model, input_sequence, max_length, SOS_token, EOS_token, output_attentions=False):
        model.eval()
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=self.device)

        for _ in range(max_length):
            
            if output_attentions:
                pred, attentions = model(input_sequence, y_input, output_attentions=output_attentions)
            else:
                pred = model(input_sequence, y_input, output_attentions=output_attentions)
            
            pred[:,:,SOS_token] = -1e20
            pred[:,:,EOS_token] = -1e20

            next_item = pred.topk(1)[1].view(-1)[-1].item()
            next_item = torch.tensor([[next_item]], device=self.device)

            y_input = torch.cat((y_input, next_item), dim=1)


        if output_attentions:
            return [i for i in y_input.view(-1).tolist()], attentions
        else:
            return [i for i in y_input.view(-1).tolist()]
    
    def post_process_prediction(self, model, prediction, empty_token):
        go_term_to_region = dict()
        if len(prediction) == 0:
            return go_term_to_region
        
        last_different_token = prediction[0]
        region_start_idx = 0
        region_end_idx = region_start_idx
        
        for idx in range(1, len(prediction)):
            current_token = prediction[idx]
            if current_token == last_different_token:
                region_end_idx += 1
            else:
                if last_different_token not in go_term_to_region:
                    go_term_to_region[last_different_token] = list()
                    
                go_term_to_region[last_different_token].append((region_start_idx, region_end_idx))
                
                
                last_different_token = current_token
                region_start_idx = region_end_idx + 1
                region_end_idx = region_start_idx
        
        if last_different_token not in go_term_to_region:
            go_term_to_region[last_different_token] = list()
                    
        go_term_to_region[last_different_token].append((region_start_idx, region_end_idx))
        
        result = dict()
        for key, value in go_term_to_region.items():
            regions = [[i[0] + 1, i[1] + 1] for i in value]
            result[model.get_config().reverse_go_term_to_index[key]] = regions
        
        return result

    def post_process_prediction_as_str(self, post_process_result: dict) -> str:
        string_io = StringIO()
        for key, value in post_process_result.items():
            string_io.write(f"{key}: {json.dumps(value)} | ")
        return string_io.getvalue()
    
    def train_loop(self, model, opt, loss_fn, dataloader, writer, total_go_term_count, trg_pad_idx, last_n=0, evaluate=False,
              empty_token=-1):
        model.train()
        total_loss = 0
        
        m = 0
        total_sample_count = 0
        
        running_metrics = {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0}

        for batch in dataloader:
            m += len(batch)
            print(m, end = ' | ')
            X, y = DatasetUtils.split_batch_into_X_y(batch)
            X, y = torch.tensor(X, dtype=torch.long).to(self.device), torch.tensor(y, dtype=torch.long).to(self.device)

            y_input = y[:,:-1]
            y_expected = y[:,1:]

            pred = model(X, y_input)
            pred = pred.permute(0, 2, 1)

            loss = loss_fn(pred, y_expected)
            print(loss)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
            total_loss += loss.detach().item()

            if evaluate:
                with torch.no_grad():
                    pred_np = pred.cpu().numpy()
                    
                    for i in range(len(pred_np)):
                        label = y[i][y[i] != trg_pad_idx][1:-1]
                        predictions = np.argmax(Utils.softmax(pred_np[i]), axis=0)[:len(label)]
                        predictions_unique = list(np.unique(np.asarray(predictions)))

                        fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(list(label.cpu().numpy()), predictions_unique, total_go_term_count)
                        precision_score, recall_score, accuracy_score = Utils.precision(tp, fp), Utils.recall(tp, fn), Utils.accuracy(fp, tp, fn, tn)
                        f1_score = Utils.f1(precision_score, recall_score)
                        
                                                
                        running_metrics["precision"] += precision_score
                        running_metrics["recall"] += recall_score
                        running_metrics["accuracy"] += accuracy_score
                        running_metrics["f1"] += f1_score
                        
                        total_sample_count += 1
                        writer.add_scalar("precision_per_sample_train_loop", precision_score, total_sample_count + last_n)
                        writer.add_scalar("recall_per_sample_train_loop", recall_score, total_sample_count + last_n)
                        writer.add_scalar("accuracy_per_sample_train_loop", accuracy_score, total_sample_count + last_n)
                        writer.add_scalar("f1_per_sample_train_loop", f1_score, total_sample_count + last_n)

        
        if evaluate:
            average_metrics = {}
            for key, value in running_metrics.items():
                average_metrics[key] = value/total_sample_count

            return total_loss / m, average_metrics, total_sample_count + last_n

        else:
            return total_loss / m
    
    def validation_loop(self, model, loss_fn, dataloader, writer, total_go_term_count, trg_pad_idx, last_n = 0):
        model.eval()
        total_loss = 0
        
        m = 0
        total_sample_count = 0

        running_metrics = {"precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0}

        with torch.no_grad():
            for batch in dataloader:
                m += len(batch)
                print(m, end = ' | ')
                X, y = DatasetUtils.split_batch_into_X_y(batch)
                X, y = torch.tensor(X, dtype=torch.long, device=self.device), torch.tensor(y, dtype=torch.long, device=self.device)

                y_input = y[:,:-1]
                y_expected = y[:,1:]
                
                pred = model(X, y_input)
                pred = pred.permute(0, 2, 1)

                pred_np = pred.cpu().numpy()
                for i in range(len(pred_np)):

                    label = y[i][y[i] != trg_pad_idx][1:-1]
                    predictions = np.argmax(Utils.softmax(pred_np[i]), axis=0)[:len(label)]
                    predictions_unique = list(np.unique(np.asarray(predictions)))

                    fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(list(label.cpu().numpy()), predictions_unique, total_go_term_count)
                    precision_score, recall_score, accuracy_score = Utils.precision(tp, fp), Utils.recall(tp, fn), Utils.accuracy(fp, tp, fn, tn)
                    f1_score = Utils.f1(precision_score, recall_score)
                    
                    running_metrics["precision"] += precision_score
                    running_metrics["recall"] += recall_score
                    running_metrics["accuracy"] += accuracy_score
                    running_metrics["f1"] += f1_score
                    
                    total_sample_count += 1
                    writer.add_scalar("precision_per_sample_validation_loop", precision_score, total_sample_count + last_n)
                    writer.add_scalar("recall_per_sample_validation_loop", recall_score, total_sample_count + last_n)
                    writer.add_scalar("accuracy_per_sample_validation_loop", accuracy_score, total_sample_count + last_n)
                    writer.add_scalar("f1_per_sample_validation_loop", f1_score, total_sample_count + last_n)
                
                loss = loss_fn(pred, y_expected)
                print(loss)
                total_loss += loss.detach().item()
        
        average_metrics = {}
        for key, value in running_metrics.items():
            average_metrics[key] = value/total_sample_count

        return total_loss / m, average_metrics, total_sample_count + last_n
    
    def fit(self,
            model,
            opt,
            loss_fn,
            train_dataloader,
            val_dataloader,
            epochs,
            writer,
            total_go_term_count,
            pred_SOS_token_id,
            pred_EOS_token_id,
            pred_EMPTY_token_id,
            pred_reverse_word_index,
            true_SOS_token_id,
            true_EOS_token_id,
            true_EMPTY_token_id,
            true_reverse_word_index,
            logger,
            evaluate_while_training=False,
            trg_pad_idx=0,
            save_per_epoch=None,
            model_save_dir=None,
            model_checkpoint_name_prefix=None):
        train_loss_list, validation_loss_list = [], []
        
        print("Training and validating model")
        last_n = 0
        last_n_training = 0
        last_n_calculate_metrics = 0
        for epoch in range(epochs):
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            
            if evaluate_while_training:
                train_loss, average_metrics, last_n_training = self.train_loop(model, opt, loss_fn, train_dataloader, writer, total_go_term_count, trg_pad_idx, last_n_training, evaluate=True)
                writer.add_scalar("precision on training set", average_metrics["precision"], epoch + 1)
                writer.add_scalar("recall on training set", average_metrics["recall"], epoch + 1)
                writer.add_scalar("f1-score on training set", average_metrics["f1"], epoch + 1)
                writer.add_scalar("accuracy on training set", average_metrics["accuracy"], epoch + 1)
            else:
                train_loss = self.train_loop(model, opt, loss_fn, train_dataloader, writer, total_go_term_count, trg_pad_idx, last_n, evaluate=False)
            
            train_loss_list += [train_loss]
            
            validation_loss, average_metrics, last_n = self.validation_loop(model, loss_fn, val_dataloader, writer, total_go_term_count, trg_pad_idx, last_n)
            validation_loss_list += [validation_loss]
            
            print(f"Training loss: {train_loss:.4f}")
            print(f"Validation loss: {validation_loss:.4f}")
            print()
            
            writer.add_scalar("training loss", train_loss, epoch + 1)
            writer.add_scalar("validation loss", validation_loss, epoch + 1)

            writer.add_scalar("precision on validation set", average_metrics["precision"], epoch + 1)
            writer.add_scalar("recall on validation set", average_metrics["recall"], epoch + 1)
            writer.add_scalar("f1-score on validation set", average_metrics["f1"], epoch + 1)
            writer.add_scalar("accuracy on validation set", average_metrics["accuracy"], epoch + 1)

            average_metrics, last_n_calculate_metrics = self.calculate_metrics(model,
                                                                               val_dataloader,
                                                                               pred_SOS_token_id,
                                                                               pred_EOS_token_id,
                                                                               pred_EMPTY_token_id,
                                                                               pred_reverse_word_index,
                                                                               true_SOS_token_id,
                                                                               true_EOS_token_id,
                                                                               true_EMPTY_token_id,
                                                                               true_reverse_word_index,
                                                                               total_go_term_count,
                                                                               writer,
                                                                               logger,
                                                                               last_n_calculate_metrics)
            for key in average_metrics.keys():
                writer.add_scalar(key, average_metrics[key], epoch + 1)
            
            writer.flush()
            
            try:
                if save_per_epoch is not None and save_per_epoch > 0 and (epoch + 1) % save_per_epoch == 0:
                    model.save(os.path.join(model_save_dir, model_checkpoint_name_prefix + "_" + "epoch_" + str(epoch + 1)))
            except Exception as e:
                print(f"Cannot save at epoch {str(epoch + 1)}: {str(e)}")
            

        return train_loss_list, validation_loss_list
    
    def calculate_metrics(self,
                          model,
                          dataloader,
                          pred_SOS_token_id,
                          pred_EOS_token_id,
                          pred_EMPTY_token_id,
                          pred_reverse_word_index,
                          true_SOS_token_id,
                          true_EOS_token_id,
                          true_EMPTY_token_id,
                          true_reverse_word_index,
                          total_go_term_count,
                          writer,
                          logger=None,
                          last_n=0,
                          sample_limit=1,
                          trg_pad_idx=0):
        
        total_sample_count = 0
        running_metrics = {"precision": 0.0,
                           "recall": 0.0,
                           "accuracy": 0.0,
                           "f1": 0.0,
                           "match_ratio": 0.0,
                           "match_ratio_nonempty": 0.0,
                           "token_based_precision": 0.0,
                           "token_based_accuracy": 0.0,
                           "token_based_recall": 0.0,
                           "token_based_f1": 0.0}
        
        with torch.no_grad():
            for batch_idx in range(len(dataloader)):
                batch = dataloader[batch_idx]
                X, y = DatasetUtils.split_batch_into_X_y(batch)
                X, y = torch.tensor(X, dtype=torch.long, device=self.device), torch.tensor(y, dtype=torch.long, device=self.device)
                for idx in range(len(X)):
                    sample_X = X[idx].unsqueeze(0)
                    sample_y = y[idx][y[idx] != trg_pad_idx][1:-1]
                    
                    prediction = self.predict(model,
                                              sample_X,
                                              SOS_token=pred_SOS_token_id,
                                              EOS_token=pred_EOS_token_id,
                                              max_length=len(sample_y)+2)[1:-1]

                    
                    unique_prediction = list(np.unique(np.asarray(prediction)))
                    
                    sample_y_str_list = Utils.convert_idx_list_to_strings(sample_y, true_reverse_word_index)
                    unique_prediction_str_list = Utils.convert_idx_list_to_strings(unique_prediction, pred_reverse_word_index)
                    prediction_str_list = Utils.convert_idx_list_to_strings(prediction[:len(sample_y)], pred_reverse_word_index)

                    fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(sample_y_str_list,
                                                           unique_prediction_str_list,
                                                           total_go_term_count,
                                                           true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(), 
                                                           pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                    precision_score, recall_score, accuracy_score = Utils.precision(tp, fp), Utils.recall(tp, fn), Utils.accuracy(fp, tp, fn, tn)
                    f1_score = Utils.f1(precision_score, recall_score)
                    
                    match_ratio_score = Utils.match_ratio(sample_y_str_list, prediction_str_list)
                    match_ratio_nonempty_score = Utils.match_ratio_nonempty(sample_y_str_list,
                                                                            prediction_str_list,
                                                                            true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())

                    # _t: abbreviation for _token
                    fp_t, tp_t, fn_t, tn_t = Utils.get_fp_tp_fn_tn_token_based(sample_y_str_list,
                                                                               prediction_str_list,
                                                                               total_go_term_count,
                                                                               true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                                               pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
                    precision_score_t, recall_score_t, accuracy_score_t = Utils.precision(tp_t, fp_t), Utils.recall(tp_t, fn_t), \
                                                                        Utils.accuracy(fp_t, tp_t, fn_t, tn_t)
                    f1_score_t = Utils.f1(precision_score_t, recall_score_t)
                    
                    
                    running_metrics["precision"] += precision_score
                    running_metrics["recall"] += recall_score
                    running_metrics["accuracy"] += accuracy_score
                    running_metrics["f1"] += f1_score
                    
                    running_metrics["match_ratio"] += match_ratio_score
                    running_metrics["match_ratio_nonempty"] += match_ratio_nonempty_score
                    running_metrics["token_based_precision"] += precision_score_t
                    running_metrics["token_based_recall"] += recall_score_t
                    running_metrics["token_based_accuracy"] += accuracy_score_t
                    running_metrics["token_based_f1"] += f1_score_t
                    
                    total_sample_count += 1
                    
                    writer.add_scalar("precision_per_sample", precision_score, total_sample_count + last_n)
                    writer.add_scalar("recall_per_sample", recall_score, total_sample_count + last_n)
                    writer.add_scalar("accuracy_per_sample", accuracy_score, total_sample_count + last_n)
                    writer.add_scalar("f1_per_sample", f1_score, total_sample_count + last_n)
                    
                    writer.add_scalar("match_ratio_per_sample", match_ratio_score, total_sample_count + last_n)
                    writer.add_scalar("match_ratio_nonempty_per_sample", match_ratio_nonempty_score, total_sample_count + last_n)
                    writer.add_scalar("token_based_precision_per_sample", precision_score_t, total_sample_count + last_n)
                    writer.add_scalar("token_based_recall_per_sample", recall_score_t, total_sample_count + last_n)
                    writer.add_scalar("token_based_accuracy_per_sample", accuracy_score_t, total_sample_count + last_n)
                    writer.add_scalar("token_based_f1_per_sample", f1_score_t, total_sample_count + last_n)
                        
                    if sample_limit is not None and total_sample_count >= sample_limit:
                        break
                
                if sample_limit is not None and total_sample_count >= sample_limit:
                    break
                    
        average_metrics = {}
        for key, value in running_metrics.items():
            average_metrics[key] = value/total_sample_count
        
        return average_metrics, total_sample_count + last_n

    def evaluate_models(self,
                        training_dataloader,
                        validation_dataloader,
                        pred_SOS_token_id,
                        pred_EOS_token_id,
                        pred_EMPTY_token_id,
                        pred_reverse_word_index,
                        true_SOS_token_id,
                        true_EOS_token_id,
                        true_EMPTY_token_id,
                        true_reverse_word_index,
                        total_go_term_count,
                        hyperparameters,
                        device,
                        src_pad_idx,
                        trg_pad_idx,
                        src_vocab_size,
                        trg_vocab_size,
                        external_encoder,
                        embed_size=1024,
                        max_length=10000,
                        epoch=10,
                        logger=None,
                        tensorboard_log_dir="runs",
                        model_save_dir=None,
                        save_per_epoch=None,
                        go_embedding_fetcher=None,
                        weights=None,
                        model_config: TransformerModelConfig=None):
        
        """
        Constructs and evaluates models based on given hyperparameters. Results are written to tensorboard log directory using SummaryWriter class.
        training_dataloader -- A list consisting of training batches
        validation_dataloader -- A list consisting of validation batches
        eos_token -- The number used for end of sentence token for GO sequence part
        total_go_term_count -- total number of go terms found in the dataset
        hyperparameters -- A dictionary where keys are hyperparameter name, e.g. lr, and values are different values for the corresponding hyperparameter.
                        Names for the hyperparameters must be same with the argument names of the functions.
        device -- The device to be used for training of the models, e.g., cpu, cuda:0, cuda:1, etc.
        src_pad_idx -- The number used as the pad index for source sequences
        trg_pad_idx -- The number used as the pad index for target sequences
        src_vocab_size -- The number of tokens found in the vocabulary constructed from source sequences
        trg_vocab_size -- The number of tokens found in the vocabulary constructed from target sequences
        external_encoder -- The external encoder object to be used as the Encoder part of the Transformer model
        embed_size -- The size of the embedding constructed by the external encoder
        max_length -- Maximum possible length of a target sequence
        epoch -- The number of epochs for which the models will be trained
        logger -- A function that logs the string given to it
        tensorboard_log_dir -- Name of the directory to which tensorboard logs will be written
        model_save_dir -- The directory to which models are to be saved
        """
        
        if model_save_dir:
            if os.path.exists(model_save_dir):
                raise RuntimeError(f"Model save directory already exists!: {model_save_dir}")
                
            os.mkdir(model_save_dir)
        
        
        if "embed_size" not in hyperparameters:
            hyperparameters["embed_size"] = [embed_size]
        
        if "max_length" not in hyperparameters:
            hyperparameters["max_length"] = [max_length]
        
        hyperparameter_option_counts = [len(value) for key, value in hyperparameters.items()]
        hyperparameter_keys = list(hyperparameters.keys())
        lookup_matrix, n = Utils.generate_lookup_matrix(hyperparameter_option_counts)
        
        for i in range(n):
            # free memory
            torch.cuda.empty_cache()
            gc.collect()
        
        
            option_indexes = lookup_matrix[i]
            
            selected_hyperparameters = {}
            lr_setting = {}
            for idx in range(len(option_indexes)):
                option_index = option_indexes[idx]
                hyperparameter_key = hyperparameter_keys[idx]
                if hyperparameter_key == "lr":
                    lr_setting["lr"] = hyperparameters["lr"][option_index]
                    continue
                selected_hyperparameters[hyperparameter_key] = hyperparameters[hyperparameter_key][option_index]
            
            
            model = ModelNavigator.create(model_config, external_encoder, device)
            
            optimizer = torch.optim.SGD(model.parameters(), **lr_setting)
            
            if weights is not None:
                loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
            else:
                loss_function = nn.CrossEntropyLoss()
                        
            selected_hyperparameters["lr"] = lr_setting["lr"]
            model_name = Utils.create_model_name_from_hyperparameters(selected_hyperparameters)
            writer = SummaryWriter(os.path.join(tensorboard_log_dir, model_name))
            
            train_loss_list, validation_loss_list = self.fit(model,
                                                             optimizer,
                                                             loss_function,
                                                             training_dataloader,
                                                             validation_dataloader,
                                                             epoch,
                                                             writer,
                                                             total_go_term_count,
                                                             pred_SOS_token_id,
                                                             pred_EOS_token_id,
                                                             pred_EMPTY_token_id,
                                                             pred_reverse_word_index,
                                                             true_SOS_token_id,
                                                             true_EOS_token_id,
                                                             true_EMPTY_token_id,
                                                             true_reverse_word_index,
                                                             logger,
                                                             True,
                                                             trg_pad_idx,
                                                             save_per_epoch=save_per_epoch,
                                                             model_save_dir=model_save_dir,
                                                             model_checkpoint_name_prefix=model_name)

            writer.flush()
            writer.close()
            
            if model_save_dir:
                model.save(os.path.join(model_save_dir, model_name + "_end_of_training"))
        
    def train(self,
              train_batches,
              validation_batches,
              total_go_term_count,
              pred_SOS_token_id,
              pred_EOS_token_id,
              pred_EMPTY_token_id,
              pred_reverse_word_index,
              true_SOS_token_id,
              true_EOS_token_id,
              true_EMPTY_token_id,
              true_reverse_word_index,
              learning_rate,
              src_pad_idx,
              trg_pad_idx,
              src_vocab_size,
              trg_vocab_size,
              prot_t5_model,
              embed_size,
              epoch,
              tensorboard_log_dir,
              model_save_dir,
              save_per_epoch,
              go_embedding_fetcher=None,
              weights=None,
              model_config:TransformerModelConfig=None):
        self.evaluate_models(train_batches,
                             validation_batches, 
                             pred_SOS_token_id,
                             pred_EOS_token_id,
                             pred_EMPTY_token_id,
                             pred_reverse_word_index,
                             true_SOS_token_id,
                             true_EOS_token_id,
                             true_EMPTY_token_id,
                             true_reverse_word_index,
                             total_go_term_count,
                             {"lr": [learning_rate]},
                             self.device,
                             src_pad_idx,
                             trg_pad_idx,
                             src_vocab_size,
                             trg_vocab_size,
                             prot_t5_model,
                             embed_size,
                             epoch=epoch,
                             tensorboard_log_dir=tensorboard_log_dir,
                             model_save_dir=model_save_dir,
                             save_per_epoch=save_per_epoch,
                             go_embedding_fetcher=go_embedding_fetcher,
                             weights=weights,
                             model_config=model_config)