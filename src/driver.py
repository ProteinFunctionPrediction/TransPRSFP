from transformers import T5EncoderModel
from transformers import T5Tokenizer
from model.model import Model
from universal.access.universal_access import UniversalAccess
from universal.settings.settings import Settings
from model.model_navigator import ModelNavigator
from utils.dataset.classification_head.dataset_converter import DatasetConverter
from utils.dataset.dataset_utils import DatasetUtils
from utils.dataset.classification_head.protein_dataset import ProteinDataset
from utils.model.classification_head.classification_head_utils import ClassificationHeadUtils
from utils.model.transformer.transformer_utils import TransformerUtils
from model.classification_head.config.classification_head_model_config import ClassificationHeadModelConfig
from model.transformer.config.transformer_model_config import TransformerModelConfig
from torch.utils.data import DataLoader
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.model.transformer.weight_setter import WeightSetter
from config.loader.json.json_config_loader import JSONConfigLoader
from utils.utils import Utils
from io import StringIO

class Driver:
    def __init__(self, args) -> None:
        self.args = args
        self.prot_t5_model: T5EncoderModel = None
        self.tokenizer: T5Tokenizer = None
        self.model: Model = None
        self.dataset: np.array = None
        self.max_length: int = None
    
    def load_models(self):
        UniversalAccess.output.write("Loading ProtT5 tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        UniversalAccess.output.write("Done!")
        
        UniversalAccess.output.write("Loading ProtT5 encoder...")
        self.prot_t5_model = T5EncoderModel.from_pretrained(self.args.prot_t5_model_path)
        for i in self.prot_t5_model.parameters():
            i.requires_grad = False
        UniversalAccess.output.write("Done!")

        model_type: str
        if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
            model_type = "classification"
        else:
            model_type = "transformer"

        if self.args.inference and self.args.model_type != Settings.MERGED_MODEL_TYPE:
            UniversalAccess.output.write(f"Loading {model_type} model...")
            self.model = ModelNavigator.load(self.args.model_path, self.prot_t5_model, self.args.device)
            UniversalAccess.output.write("Done!")

    def run(self):
        self.load_models()

        self.dataset = DatasetUtils.load(self.args.dataset)
        if self.args.train:
            if self.args.max_length > 0:
                self.max_length = self.args.max_length
            else:
                self.max_length = DatasetUtils.infer_maximum_length(self.dataset)
                if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE and self.max_length < 10000:
                    self.max_length = 10000
        elif self.model is not None:
            self.max_length = self.model.config.max_length
        
        
        if self.args.train:
            BATCH_SIZE = self.args.batch_size
        else:
            BATCH_SIZE = 8
        
        if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:
            dataset_converter = DatasetConverter(self.dataset, ignore_tokens=[Settings.TRANSFORMER_OOV_TOKEN, Settings.TRANSFORMER_SOS_TOKEN, Settings.TRANSFORMER_EOS_TOKEN])
            dataset_converter.convert()
            dataset_converted = dataset_converter.get_new_dataset()
        
            classification_head_utils = ClassificationHeadUtils(self.args.device)
        
        if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:

            transformer_utils = TransformerUtils(self.args.device)
            
        
        if self.args.inference:
            if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:
                
                if self.args.model_type == Settings.MERGED_MODEL_TYPE:
                    self.max_length = self.get_value_from_config_of_model(self.args.classification_head_model_path, "max_length")
                
                data = ProteinDataset(dataset_converted, self.tokenizer, self.max_length)
                data_loader = DataLoader(data, batch_size=BATCH_SIZE)
            
            if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:        
                if self.args.model_type == Settings.MERGED_MODEL_TYPE:
                    self.max_length = self.get_value_from_config_of_model(self.args.transformer_model_path, "max_length")
            
                batches = list(DatasetUtils.generate_batch_iterator(self.dataset, self.tokenizer, None, BATCH_SIZE))
            
            if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
                predictions = classification_head_utils.run_classification_head_prediction(data_loader, self.model, self.args.threshold, BATCH_SIZE, self)
            
            elif self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE:
                predictions = transformer_utils.run_transformer_prediction(batches, self.model, BATCH_SIZE, self)
            
            else: # merged model
                UniversalAccess.output.write(f"Loading classification model...")
                self.model = ModelNavigator.load(self.args.classification_head_model_path, self.prot_t5_model, self.args.device)
                UniversalAccess.output.write("Done!")
                self.max_length = self.model.config.max_length
                classification_model_go_term_to_index = self.model.config.go_term_to_index
                classification_model_reverse_go_term_to_index = self.model.config.reverse_go_term_to_index
                classification_predictions = classification_head_utils.run_classification_head_prediction(data_loader, self.model, self.args.threshold, BATCH_SIZE, self)
                
                Utils.free_gpu_memory()
                
                UniversalAccess.output.write(f"Loading transformer model...")
                self.model = ModelNavigator.load(self.args.transformer_model_path, self.prot_t5_model, self.args.device)
                UniversalAccess.output.write("Done!")
                self.max_length = self.model.config.max_length
                transformer_model_go_term_to_index = self.model.config.go_term_to_index
                transformer_model_reverse_go_term_to_index = self.model.config.reverse_go_term_to_index
                transformer_predictions = transformer_utils.run_transformer_prediction(batches, self.model, BATCH_SIZE, self)
                
                self.produce_merged_prediction_output(classification_predictions, transformer_predictions,
                                                      classification_model_go_term_to_index,
                                                      classification_model_reverse_go_term_to_index,
                                                      transformer_model_go_term_to_index,
                                                      transformer_model_reverse_go_term_to_index,
                                                      transformer_utils)
                

        else: # training mode
            
            if not os.path.exists(self.args.model_save_dir) and self.args.model_type != Settings.TRANSFORMER_MODEL_TYPE:
                os.mkdir(self.args.model_save_dir)
            
            if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
                n_train = DatasetUtils.get_training_count(self.args.training_dataset_ratio, len(dataset_converted))
                
                
                train_data = ProteinDataset(dataset_converted[:n_train], self.tokenizer, self.max_length)
                train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
                
                val_data = ProteinDataset(dataset_converted[n_train:], self.tokenizer, self.max_length)
                val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
                
                go_term_count = DatasetUtils.get_go_term_count(self.dataset)
                model_config = ClassificationHeadModelConfig("./model.pth", go_term_count,
                                                             self.max_length, "./go_term_to_index.pkl",
                                                             dataset_converter.go_term_to_index_map)
                model_config.build()
                self.model = ModelNavigator.create(model_config, self.prot_t5_model, self.args.device)
                
                classification_head_utils.train(self.args.epoch, self.args.learning_rate, self.model,
                                                go_term_count, train_loader, val_loader,
                                                self.args.save_per_epoch, self.args.model_save_dir)
            
            elif self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE:
                n_train = DatasetUtils.get_training_count(self.args.training_dataset_ratio, len(self.dataset))
                
                tf_tokenizer = Tokenizer(oov_token='<OOV>', filters='')
                tf_tokenizer.fit_on_texts(self.dataset[:, 1])
                
                weight_setter = WeightSetter(self.dataset, tf_tokenizer, len(tf_tokenizer.word_index) + 1, Settings.TRANSFORMER_TRG_PAD_IDX)
                weights = weight_setter.get_weights()
                
                train_batches = list(DatasetUtils.generate_batch_iterator(self.dataset[:n_train], self.tokenizer, tf_tokenizer, BATCH_SIZE))
                validation_batches = list(DatasetUtils.generate_batch_iterator(self.dataset[n_train:], self.tokenizer, tf_tokenizer, BATCH_SIZE))
                
                model_config = TransformerModelConfig("./model.pth", len(self.tokenizer.get_vocab()),
                                                      len(tf_tokenizer.word_index) + 1,
                                                      tf_tokenizer.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                      tf_tokenizer.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                      self.max_length, Settings.TRANSFORMER_EMBED_SIZE,
                                                      Settings.TRANSFORMER_NUM_LAYERS, Settings.TRANSFORMER_HEADS,
                                                      "./go_term_to_index.pkl", tf_tokenizer.word_index)
                model_config.build()

                transformer_utils.train(train_batches, validation_batches,
                                        tf_tokenizer, self.args.learning_rate,
                                        Settings.TRANSFORMER_SRC_PAD_IDX,
                                        Settings.TRANSFORMER_TRG_PAD_IDX,
                                        model_config.src_vocab_size,
                                        model_config.trg_vocab_size,
                                        self.prot_t5_model, model_config.embed_size,
                                        self.args.epoch, self.args.tensorboard_log_dir,
                                        self.args.model_save_dir, self.args.save_per_epoch,
                                        None, weights, model_config)
                
    
    def notify(self, index: int, prediction: str) -> None:
        protein_sequence = self.dataset[index][0].replace(" ", "")
        UniversalAccess.output.write(f"{protein_sequence}: {prediction}")

    def get_value_from_config_of_model(self, folder_path:str, key: str) -> any:
        config_filepath = os.path.join(folder_path, Settings.CONFIG_FILENAME)
        json_config_loader = JSONConfigLoader(config_filepath)
        return json_config_loader.config[key]
    
    def produce_merged_prediction_output(self, classification_predictions, transformer_predictions,
                                                classification_model_go_term_to_index,
                                                classification_model_reverse_go_term_to_index,
                                                transformer_model_go_term_to_index,
                                                transformer_model_reverse_go_term_to_index,
                                                transformer_utils: TransformerUtils):
        for idx in range(len(classification_predictions)):
            protein_sequence = self.dataset[idx][0].replace(" ", "")
            
            classification_prediction = classification_predictions[idx]
            transformer_prediction = transformer_predictions[idx]
            
            merged_predictions_str = self.merge_predictions(classification_prediction, transformer_prediction,
                classification_model_go_term_to_index,
                classification_model_reverse_go_term_to_index,
                transformer_model_go_term_to_index,
                transformer_model_reverse_go_term_to_index,
                transformer_utils)
            
            UniversalAccess.output.write(f"{protein_sequence}: {merged_predictions_str}")
    
    def merge_predictions(self, classification_prediction, transformer_prediction,
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
                transformer_utils.post_process_prediction(self.model, transformer_prediction,
                                                      self.model.config.go_term_to_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None))
            )
        )
        
        return string_io.getvalue()
