from io import StringIO

from universal.access.universal_access import UniversalAccess
from universal.settings.settings import Settings
from utils.model.transformer.transformer_utils import TransformerUtils
from utils.utils import Utils


class DriverUtils:
    @staticmethod
    def merge_predictions(driver,
                          classification_prediction,
                          transformer_prediction,
                          classification_model_go_term_to_index,
                          classification_model_reverse_go_term_to_index,
                          transformer_model_go_term_to_index,
                          transformer_model_reverse_go_term_to_index,
                          transformer_utils: TransformerUtils):
        transformer_prediction_unique = list(set(transformer_prediction))
        transformer_prediction_unique_str_list = Utils.convert_tokens_to_str_go_terms(
            transformer_model_reverse_go_term_to_index, transformer_prediction_unique)
        
        classification_prediction_str_list = Utils.convert_tokens_to_str_go_terms(
            classification_model_reverse_go_term_to_index, classification_prediction)

        merged_str_list = list(set(transformer_prediction_unique_str_list + classification_prediction_str_list))
        
        string_io = StringIO()
        string_io.write(" ".join(merged_str_list) + ": ")
        
        string_io.write(
            transformer_utils.post_process_prediction_as_str(
                transformer_utils.post_process_prediction(driver.model, transformer_prediction,
                                                      driver.model.get_config().go_term_to_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None))
            )
        )

        return string_io.getvalue()

    @staticmethod
    def produce_merged_prediction_output(driver, classification_predictions, transformer_predictions,
                                                classification_model_go_term_to_index,
                                                classification_model_reverse_go_term_to_index,
                                                transformer_model_go_term_to_index,
                                                transformer_model_reverse_go_term_to_index,
                                                transformer_utils: TransformerUtils):
        for idx in range(len(classification_predictions)):
            protein_sequence = driver.dataset[idx][0].replace(" ", "")
            
            classification_prediction = classification_predictions[idx]
            transformer_prediction = transformer_predictions[idx]
            
            merged_predictions_str = DriverUtils.merge_predictions(driver,
                classification_prediction, transformer_prediction,
                classification_model_go_term_to_index,
                classification_model_reverse_go_term_to_index,
                transformer_model_go_term_to_index,
                transformer_model_reverse_go_term_to_index,
                transformer_utils)
            
            UniversalAccess.output.write(f"{protein_sequence}: {merged_predictions_str}")
    

    @staticmethod
    def evaluate_merged_mode_prediction(driver,
                                        classification_predictions,
                                        transformer_predictions,
                                        classification_model_go_term_to_index,
                                        classification_model_reverse_go_term_to_index,
                                        transformer_model_go_term_to_index,
                                        transformer_model_reverse_go_term_to_index):
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        for idx in range(len(classification_predictions)):
            protein_sequence = driver.dataset[idx][0].replace(" ", "")
            
            classification_prediction = classification_predictions[idx]
            transformer_prediction = transformer_predictions[idx]
            
            merged_prediction = set()
            
            for predicted_token in classification_prediction:
                merged_prediction.add(classification_model_reverse_go_term_to_index[predicted_token].upper())
            
            for predicted_token in transformer_prediction:
                if predicted_token != Settings.TRANSFORMER_EMPTY_TOKEN:
                    merged_prediction.add(transformer_model_reverse_go_term_to_index[predicted_token].upper())
            
            merged_prediction = list(merged_prediction)

            groundtruth_tokens = driver.dataset[idx][1].split()[1:-1] # exclude <sos> and <eos> tokens at the beginning and the end
            fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(groundtruth_tokens,
                                                   merged_prediction,
                                                   0,
                                                   pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                   true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
            precision_score, recall_score = Utils.precision(tp, fp), Utils.recall(tp, fn)
            f1_score = Utils.f1(precision_score, recall_score)
            
            metrics["precision"] += precision_score
            metrics["recall"] += recall_score
            metrics["f1"] += f1_score
        
        metrics["precision"] = metrics["precision"] / len(classification_predictions)
        metrics["recall"] = metrics["recall"] / len(classification_predictions)
        metrics["f1"] = metrics["f1"] / len(classification_predictions)
        
        return metrics
    
    def evaluate_classification_head_predictions(driver,
                                                 classification_predictions,
                                                 classification_model_go_term_to_index,
                                                 classification_model_reverse_go_term_to_index):
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        for idx in range(len(classification_predictions)):
            protein_sequence = driver.dataset[idx][0].replace(" ", "")
            
            classification_prediction = classification_predictions[idx]
            
            merged_prediction = set()
            
            for predicted_token in classification_prediction:
                merged_prediction.add(classification_model_reverse_go_term_to_index[predicted_token].upper())

            merged_prediction = list(merged_prediction)

            groundtruth_tokens = driver.dataset[idx][1].split()[1:-1] # exclude <sos> and <eos> tokens at the beginning and the end
            fp, tp, fn, tn = Utils.get_fp_tp_fn_tn(groundtruth_tokens,
                                                   merged_prediction,
                                                   0,
                                                   true_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper(),
                                                   pred_empty_token=Settings.TRANSFORMER_EMPTY_TOKEN.upper())
            precision_score, recall_score = Utils.precision(tp, fp), Utils.recall(tp, fn)
            f1_score = Utils.f1(precision_score, recall_score)
            
            metrics["precision"] += precision_score
            metrics["recall"] += recall_score
            metrics["f1"] += f1_score
    
        metrics["precision"] = metrics["precision"] / len(classification_predictions)
        metrics["recall"] = metrics["recall"] / len(classification_predictions)
        metrics["f1"] = metrics["f1"] / len(classification_predictions)
        
        return metrics
