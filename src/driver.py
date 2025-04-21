from transformers import T5EncoderModel
from transformers import T5Tokenizer
from transformers import TrainingArguments
from transformers import GPT2LMHeadModel
from transformers import GPT2Config
import evaluate
from model.model import Model
from universal.access.universal_access import UniversalAccess
from universal.settings.settings import Settings
from model.model_navigator import ModelNavigator
from utils.dataset.classification_head.dataset_converter import DatasetConverter
from utils.dataset.dataset_utils import DatasetUtils
from utils.dataset.classification_head.protein_dataset import ProteinDataset
from utils.model.classification_head.classification_head_utils import ClassificationHeadUtils
from utils.model.transformer.transformer_utils import TransformerUtils
from utils.dataset.gpt2_lmhead.dataset_utils import DatasetUtils as GPT2DatasetUtils
from utils.model.gpt2_lmhead.trainer import GPT2LMHeadTrainer
from utils.model.gpt2_lmhead.gpt2_lmhead_utils import GPT2LMHeadUtils
from utils.dataset.gpt2_lmhead.dataset import GPT2Dataset
from model.classification_head.config.classification_head_model_config import ClassificationHeadModelConfig
from model.transformer.config.transformer_model_config import TransformerModelConfig
from model.gpt2_lmhead.config.gpt2_lmhead_model_config import Gpt2LMHeadModelConfig
import pickle

from torch.utils.data import DataLoader
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.model.transformer.weight_setter import WeightSetter
from config.loader.json.json_config_loader import JSONConfigLoader
from utils.utils import Utils
from io import StringIO
import torch

class Driver:
    def __init__(self, args) -> None:
        self.args = args
        self.prot_t5_model: T5EncoderModel = None
        self.tokenizer: T5Tokenizer = None
        self.model: Model = None
        self.dataset: np.array = None
        self.max_length: int = None
        self.accuracy_metric = evaluate.load("accuracy")
        self.empty_token_id = None
        self.sos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.total_go_term_count = None
        self.loaded_go_term_index = None
        self.reverse_loaded_go_term_index = None
        self.go_term_metrics_filepath_prefix = self.args.go_term_metrics_filepath_prefix
        if self.go_term_metrics_filepath_prefix == '' or self.go_term_metrics_filepath_prefix is None:
            self.go_term_metrics_filepath_prefix = Utils.generate_random_str(length=32, use_timestamp=True)

        if self.args.load_go_term_index is not None and os.path.exists(self.args.load_go_term_index):
            with open(self.args.load_go_term_index, "rb") as f:
                self.loaded_go_term_index = pickle.load(f)
                self.reverse_loaded_go_term_index = Utils.build_reverse_index(self.loaded_go_term_index)
        
        self.tf_tokenizer_fit_on_dataset = None
        self.tf_tokenizer_fit_on_dataset_reverse_word_index = None
        
        self.tf_tokenizer_pred = Tokenizer(oov_token='<OOV>', filters='')
        self.tf_tokenizer_pred_reverse_word_index = None
        if self.loaded_go_term_index is not None:
            self.tf_tokenizer_pred.word_index = self.loaded_go_term_index
            self.tf_tokenizer_pred_reverse_word_index = self.reverse_loaded_go_term_index
        self.model_go_term_index = None
        self.reverse_model_go_term_index = None


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

        if self.args.inference and self.args.model_type != Settings.MERGED_MODEL_TYPE and self.args.model_type != Settings.GPT2_MODEL_TYPE:
            UniversalAccess.output.write(f"Loading {model_type} model...")
            self.model = ModelNavigator.load(self.args.model_path, self.prot_t5_model, self.args.device)
            self.model_go_term_index = self.model.get_config().go_term_to_index
            self.reverse_model_go_term_index = self.model.get_config().reverse_go_term_to_index
            if self.loaded_go_term_index is None:
                self.tf_tokenizer_pred.word_index = self.model_go_term_index
                self.tf_tokenizer_pred_reverse_word_index = self.reverse_model_go_term_index
            UniversalAccess.output.write("Done!")
        elif self.args.inference and self.args.model_type == Settings.GPT2_MODEL_TYPE:
            UniversalAccess.output.write(f"Loading {model_type} model...")
            gpt2_lmhead_pretrained_config = GPT2Config.from_pretrained(self.args.model_path)
            self.model = GPT2LMHeadModel.from_pretrained(self.args.model_path, config=gpt2_lmhead_pretrained_config).to(self.args.device)
            self.model_go_term_index = self.loaded_go_term_index
            self.reverse_model_go_term_index = Utils.build_reverse_index(self.model_go_term_index)
            UniversalAccess.output.write("Done!")
        elif self.args.inference and self.args.model_type == Settings.MERGED_MODEL_TYPE and self.args.transformer_model_type == Settings.TRANSFORMER_MODEL_TYPE:
            transformer_model_config = ModelNavigator.load_config(self.args.transformer_model_path)
            if self.loaded_go_term_index is None:
                self.tf_tokenizer_pred.word_index = transformer_model_config.go_term_to_index
                self.tf_tokenizer_pred_reverse_word_index = Utils.build_reverse_index(self.tf_tokenizer_pred.word_index)

    def run(self):
        self.load_models()

        self.dataset = DatasetUtils.load(self.args.dataset)

        if self.args.model_type == Settings.MERGED_MODEL_TYPE or self.args.model_type == Settings.GPT2_MODEL_TYPE or self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE:
            self.tf_tokenizer_fit_on_dataset = Tokenizer(oov_token='<OOV>', filters='')
            self.tf_tokenizer_fit_on_dataset.fit_on_texts(self.dataset[:, 1])
            self.tf_tokenizer_fit_on_dataset_reverse_word_index = Utils.build_reverse_index(self.tf_tokenizer_fit_on_dataset.word_index)

        if self.tf_tokenizer_pred_reverse_word_index is None and self.tf_tokenizer_fit_on_dataset is not None:
            self.tf_tokenizer_pred.word_index = self.tf_tokenizer_fit_on_dataset.word_index
            self.tf_tokenizer_pred_reverse_word_index = Utils.build_reverse_index(self.tf_tokenizer_pred.word_index)

        if self.args.train:
            if self.args.max_length > 0:
                self.max_length = self.args.max_length
            else:
                self.max_length = DatasetUtils.infer_maximum_length(self.dataset)
                if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE and self.max_length < 10000:
                    self.max_length = 10000
        elif self.model is not None and self.args.model_type != Settings.GPT2_MODEL_TYPE:
            self.max_length = self.model.get_config().max_length
        elif self.args.model_type == Settings.GPT2_MODEL_TYPE:
            if self.args.max_length > 0:
                self.max_length = self.args.max_length
            else:
                self.max_length = DatasetUtils.infer_maximum_length(self.dataset)


        if self.args.train:
            self.BATCH_SIZE = self.args.batch_size
        else:
            self.BATCH_SIZE = 8

        if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:
            classification_head_model_config = None
            if self.args.inference:
                if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
                    classification_head_model_config = ModelNavigator.load_config(self.args.model_path)
                elif self.args.model_type == Settings.MERGED_MODEL_TYPE:
                    classification_head_model_config = ModelNavigator.load_config(self.args.classification_head_model_path)

            go_term_to_index = None
            if classification_head_model_config is not None:
                go_term_to_index = classification_head_model_config.go_term_to_index

            dataset_converter = DatasetConverter(self.dataset, ignore_tokens=[Settings.TRANSFORMER_OOV_TOKEN, Settings.TRANSFORMER_SOS_TOKEN, Settings.TRANSFORMER_EOS_TOKEN])
            dataset_converter.convert()
            dataset_converted = dataset_converter.get_new_dataset()

            classification_head_utils = ClassificationHeadUtils(self.args.device)

        if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:
            transformer_utils = TransformerUtils(self.args.device)
            if self.args.transformer_model_type == Settings.GPT2_MODEL_TYPE:
                self.gpt2_lmhead_utils = GPT2LMHeadUtils(self.args.device)

        elif self.args.model_type == Settings.GPT2_MODEL_TYPE:
            self.gpt2_lmhead_utils = GPT2LMHeadUtils(self.args.device)


        if self.args.inference:
            if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:

                if self.args.model_type == Settings.MERGED_MODEL_TYPE:
                    self.max_length = self.get_value_from_config_of_model(self.args.classification_head_model_path, "max_length")

                data = ProteinDataset(dataset_converted, self.tokenizer, self.max_length)
                data_loader = DataLoader(data, batch_size=self.BATCH_SIZE)

            if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE:        
                if self.args.model_type == Settings.MERGED_MODEL_TYPE and self.args.transformer_model_type == Settings.TRANSFORMER_MODEL_TYPE:
                    self.max_length = self.get_value_from_config_of_model(self.args.transformer_model_path, "max_length")
                elif self.args.model_type == Settings.MERGED_MODEL_TYPE and self.args.transformer_model_type == Settings.GPT2_MODEL_TYPE:
                    self.max_length = self.args.max_length

                if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE:
                    batches = list(DatasetUtils.generate_batch_iterator(self.dataset, self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE))
                elif self.args.model_type == Settings.MERGED_MODEL_TYPE:
                    if self.args.transformer_model_type == Settings.TRANSFORMER_MODEL_TYPE:
                        batches = list(DatasetUtils.generate_batch_iterator(self.dataset, self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE))
                    elif self.args.transformer_model_type == Settings.GPT2_MODEL_TYPE:
                        batches = list(GPT2DatasetUtils.generate_torch_dataset_compatible_dataset_iterator(self.dataset, self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE, self.max_length))

            if self.args.model_type == Settings.GPT2_MODEL_TYPE:
                batches = list(GPT2DatasetUtils.generate_torch_dataset_compatible_dataset_iterator(self.dataset, self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE, self.max_length))

            if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
                predictions = classification_head_utils.run_classification_head_prediction(data_loader, self.model, self.args.threshold, self.BATCH_SIZE, self)
                if self.args.compute_metrics:
                    print(self.evaluate_classification_head_predictions(predictions, self.model.get_config().go_term_to_index, self.model.get_config().reverse_go_term_to_index))

            elif self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE:
                average_metrics, predictions = transformer_utils.run_transformer_prediction(batches,
                                                                                            self.model,
                                                                                            self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                                                            self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                                                            self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                                                            self.tf_tokenizer_pred_reverse_word_index,
                                                                                            self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_SOS_TOKEN, None),
                                                                                            self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EOS_TOKEN, None),
                                                                                            self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None),
                                                                                            self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                                                                            self.BATCH_SIZE,
                                                                                            self,
                                                                                            compute_go_based_metrics=self.args.compute_metrics,
                                                                                            save_go_based_metrics=self.args.compute_metrics,
                                                                                            compute_metrics=self.args.compute_metrics,
                                                                                            go_based_metrics_filepath=self.go_term_metrics_filepath_prefix + ".pkl")
                if self.args.compute_metrics:
                    print(average_metrics)

            elif self.args.model_type == Settings.GPT2_MODEL_TYPE:
                average_metrics, predictions = self.gpt2_lmhead_utils.run_prediction(batches=batches,
                                                                                     encoder=self.prot_t5_model.to(self.args.device),
                                                                                     model=self.model,
                                                                                     caller=self,
                                                                                     pred_SOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                                                     pred_EOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                                                     pred_EMPTY_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                                                     pred_reverse_word_index=self.tf_tokenizer_pred_reverse_word_index,
                                                                                     true_SOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_SOS_TOKEN, None),
                                                                                     true_EOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EOS_TOKEN, None),
                                                                                     true_EMPTY_token_id=self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None),
                                                                                     true_reverse_word_index=self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                                                                     compute_go_based_metrics=self.args.compute_metrics,
                                                                                     save_go_based_metrics=self.args.compute_metrics,
                                                                                     compute_metrics=self.args.compute_metrics,
                                                                                     go_based_metrics_filepath=self.go_term_metrics_filepath_prefix + ".pkl")
                if self.args.compute_metrics:
                    print(average_metrics)

            else: # merged model
                UniversalAccess.output.write(f"Loading {self.args.transformer_model_type} model...")
                if self.args.transformer_model_type == Settings.TRANSFORMER_MODEL_TYPE:
                    self.model = ModelNavigator.load(self.args.transformer_model_path, self.prot_t5_model, self.args.device)
                    UniversalAccess.output.write("Done!")
                    self.max_length = self.model.get_config().max_length
                    transformer_model_go_term_to_index = self.model.get_config().go_term_to_index
                    transformer_model_reverse_go_term_to_index = self.model.get_config().reverse_go_term_to_index
                    _, transformer_predictions = transformer_utils.run_transformer_prediction(batches,
                                                                                              self.model,
                                                                                              self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                                                              self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                                                              self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                                                              self.tf_tokenizer_pred_reverse_word_index,
                                                                                              self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_SOS_TOKEN, None),
                                                                                              self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EOS_TOKEN, None),
                                                                                              self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None),
                                                                                              self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                                                                              self.BATCH_SIZE,
                                                                                              self,
                                                                                              compute_go_based_metrics=False,
                                                                                              save_go_based_metrics=False,
                                                                                              compute_metrics=False,
                                                                                              go_based_metrics_filepath='')

                elif self.args.transformer_model_type == Settings.GPT2_MODEL_TYPE:
                    gpt2_lmhead_pretrained_config = GPT2Config.from_pretrained(self.args.transformer_model_path)
                    self.model = GPT2LMHeadModel.from_pretrained(self.args.transformer_model_path, config=gpt2_lmhead_pretrained_config).to(self.args.device)
                    UniversalAccess.output.write("Done!")
                    self.max_length = self.args.max_length

                    _, transformer_predictions = self.gpt2_lmhead_utils.run_prediction(batches=batches,
                                                                                       encoder=self.prot_t5_model.to(self.args.device),
                                                                                       model=self.model,
                                                                                       caller=self,
                                                                                       pred_SOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                                                       pred_EOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                                                       pred_EMPTY_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                                                       pred_reverse_word_index=self.tf_tokenizer_pred_reverse_word_index,
                                                                                       true_SOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_SOS_TOKEN, None),
                                                                                       true_EOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EOS_TOKEN, None),
                                                                                       true_EMPTY_token_id=self.tf_tokenizer_fit_on_dataset.word_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None),
                                                                                       true_reverse_word_index=self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                                                                       compute_go_based_metrics=False,
                                                                                       save_go_based_metrics=False,
                                                                                       go_based_metrics_filepath='',
                                                                                       compute_metrics=False)
                Utils.free_gpu_memory()
                UniversalAccess.output.write(f"Loading classification model...")
                self.model = ModelNavigator.load(self.args.classification_head_model_path, self.prot_t5_model, self.args.device)
                UniversalAccess.output.write("Done!")
                self.max_length = self.model.get_config().max_length
                classification_model_go_term_to_index = self.model.get_config().go_term_to_index
                classification_model_reverse_go_term_to_index = self.model.get_config().reverse_go_term_to_index
                classification_predictions = classification_head_utils.run_classification_head_prediction(data_loader, self.model, self.args.threshold, self.BATCH_SIZE, self)
                
                
                
                self.produce_merged_prediction_output(classification_predictions, transformer_predictions,
                                                      classification_model_go_term_to_index,
                                                      classification_model_reverse_go_term_to_index,
                                                      self.tf_tokenizer_pred.word_index,
                                                      self.tf_tokenizer_pred_reverse_word_index,
                                                      transformer_utils)
                
                if self.args.compute_metrics:
                    print(self.evaluate_merged_mode_prediction(classification_predictions,
                                                               transformer_predictions,
                                                               classification_model_go_term_to_index,
                                                               classification_model_reverse_go_term_to_index,
                                                               self.tf_tokenizer_pred.word_index,
                                                               self.tf_tokenizer_pred_reverse_word_index))

        else: # training mode
            
            if not os.path.exists(self.args.model_save_dir) and self.args.model_type != Settings.TRANSFORMER_MODEL_TYPE:
                os.mkdir(self.args.model_save_dir)
            
            if self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE:
                n_train = DatasetUtils.get_training_count(self.args.training_dataset_ratio, len(dataset_converted))
                
                
                train_data = ProteinDataset(dataset_converted[:n_train], self.tokenizer, self.max_length)
                train_loader = DataLoader(train_data, batch_size=self.BATCH_SIZE)
                
                val_data = ProteinDataset(dataset_converted[n_train:], self.tokenizer, self.max_length)
                val_loader = DataLoader(val_data, batch_size=self.BATCH_SIZE)
                
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
                
                weight_setter = WeightSetter(self.dataset,
                                             self.tf_tokenizer_fit_on_dataset,
                                             len(self.tf_tokenizer_fit_on_dataset.word_index) + 1,
                                             Settings.TRANSFORMER_TRG_PAD_IDX)
                weights = weight_setter.get_weights()
                
                train_batches = list(DatasetUtils.generate_batch_iterator(self.dataset[:n_train], self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE))
                validation_batches = list(DatasetUtils.generate_batch_iterator(self.dataset[n_train:], self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE))
                
                model_config = TransformerModelConfig("./model.pth", len(self.tokenizer.get_vocab()),
                                                      len(self.tf_tokenizer_fit_on_dataset.word_index) + 1,
                                                      self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                      self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                      self.max_length, Settings.TRANSFORMER_EMBED_SIZE,
                                                      Settings.TRANSFORMER_NUM_LAYERS, Settings.TRANSFORMER_HEADS,
                                                      "./go_term_to_index.pkl", self.tf_tokenizer_fit_on_dataset.word_index)
                model_config.build()

                transformer_utils.train(train_batches,
                                        validation_batches,
                                        len(self.tf_tokenizer_fit_on_dataset.word_index) - 1,
                                        self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                        self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                        self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                        self.tf_tokenizer_pred_reverse_word_index,
                                        self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                        self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                        self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                        self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                        self.args.learning_rate,
                                        Settings.TRANSFORMER_SRC_PAD_IDX,
                                        Settings.TRANSFORMER_TRG_PAD_IDX,
                                        model_config.src_vocab_size,
                                        model_config.trg_vocab_size,
                                        self.prot_t5_model,
                                        model_config.embed_size,
                                        self.args.epoch,
                                        self.args.tensorboard_log_dir,
                                        self.args.model_save_dir,
                                        self.args.save_per_epoch,
                                        None,
                                        weights,
                                        model_config)

            elif self.args.model_type == Settings.GPT2_MODEL_TYPE:
                n_train = DatasetUtils.get_training_count(self.args.training_dataset_ratio, len(self.dataset))

                if Settings.GPT2_USE_CUSTOM_WEIGHTS:
                    weight_setter = WeightSetter(self.dataset, self.tf_tokenizer_fit_on_dataset, len(self.tf_tokenizer_fit_on_dataset.word_index) + 1, Settings.TRANSFORMER_TRG_PAD_IDX)
                    weights = torch.tensor(weight_setter.get_weights()).to(self.args.device)
                else:
                    weights = None
                
                np.random.shuffle(self.dataset)
                train, val = DatasetUtils.split_train_val(self.dataset, len(self.dataset) - n_train)
                self._train_set = train
                self._validation_set = val
                
                if self.args.save_training_set_to is not None and not os.path.exists(self.args.save_training_set_to):
                    with open(self.args.save_training_set_to, "wb") as training_split_f:
                        pickle.dump(train, training_split_f)

                if self.args.save_test_set_to is not None and not os.path.exists(self.args.save_test_set_to):
                    with open(self.args.save_test_set_to, "wb") as validation_split_f:
                        pickle.dump(val, validation_split_f)
                
                train_batches = list(GPT2DatasetUtils.generate_torch_dataset_compatible_dataset_iterator(train, self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE, self.max_length))
                validation_batches = list(GPT2DatasetUtils.generate_torch_dataset_compatible_dataset_iterator(val, self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE, self.max_length))

                if self.args.continue_from_checkpoint is not None:
                    assert os.path.exists(self.args.continue_from_checkpoint)
                    UniversalAccess.output.write(f"Checkpoint: {self.args.continue_from_checkpoint}")
                    gpt2_lmhead_pretrained_config = GPT2Config.from_pretrained(self.args.continue_from_checkpoint)
                    model = GPT2LMHeadModel.from_pretrained(self.args.continue_from_checkpoint, config=gpt2_lmhead_pretrained_config).to(self.args.device)
                    self.model_for_inference = model
                else:
                    model_config = Gpt2LMHeadModelConfig(filepath="./model.pth",
                                                     n_embd=Settings.TRANSFORMER_EMBED_SIZE,
                                                     heads=4,
                                                     vocab_size=len(self.tf_tokenizer_fit_on_dataset.word_index) + 1,
                                                     n_positions=self.max_length,
                                                     num_layers=1,
                                                     sos_token=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                     eos_token=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                     go_term_to_index_filepath="./go_term_to_index.pkl",
                                                     go_term_to_index=self.tf_tokenizer_fit_on_dataset.word_index)
                    model_config.build()
                    model = ModelNavigator.create(config=model_config, prot_t5_model=self.prot_t5_model, device=self.args.device)
                    self.model_for_inference = model
                
                if self.args.go_term_index is not None:
                    with open(self.args.go_term_index, "wb") as go_term_index_f:
                        pickle.dump(self.tf_tokenizer_fit_on_dataset.word_index, go_term_index_f)
                
                if self.args.device == 'cpu':
                    training_args = TrainingArguments(
                        run_name=self.args.run_name,
                        output_dir=self.args.model_save_dir,
                        overwrite_output_dir=False,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        num_train_epochs=self.args.epoch,
                        learning_rate=self.args.learning_rate,
                        per_device_train_batch_size=self.BATCH_SIZE,
                        per_device_eval_batch_size=self.BATCH_SIZE,
                        save_total_limit=1,
                        disable_tqdm=True,
                        logging_steps=10,
                        dataloader_num_workers=10,
                        fp16=False,
                        ddp_find_unused_parameters=False,
                        remove_unused_columns=False,
                        eval_accumulation_steps=10,
                        no_cuda=True)
                else:
                    training_args = TrainingArguments(
                        run_name=self.args.run_name,
                        output_dir=self.args.model_save_dir,
                        overwrite_output_dir=False,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        num_train_epochs=self.args.epoch,
                        learning_rate=self.args.learning_rate,
                        per_device_train_batch_size=self.BATCH_SIZE,
                        per_device_eval_batch_size=self.BATCH_SIZE,
                        save_total_limit=1,
                        disable_tqdm=True,
                        logging_steps=10,
                        dataloader_num_workers=10,
                        fp16=False,
                        ddp_find_unused_parameters=False,
                        remove_unused_columns=False,
                        eval_accumulation_steps=10)

                
                
                trainer = GPT2LMHeadTrainer(encoder_model=self.prot_t5_model.to(self.args.device),
                                            custom_weights=weights,
                                            model=model.to(self.args.device),
                                            args=training_args,
                                            train_dataset=GPT2Dataset(train_batches),
                                            eval_dataset=GPT2Dataset(validation_batches),
                                            compute_metrics=self.compute_metrics)
                
                trainer.args._n_gpu = 1
                trainer.train()
    
    def truncate_padding(self, pred, label, pad_idx):
        if len(label) < 1 or label[0] == pad_idx:
            return [], []

        if label[-1] != pad_idx:
            return pred, label

        truncate_to = len(label)
        while truncate_to > 0 and label[truncate_to - 1] == pad_idx:
            truncate_to -= 1
        
        return pred[:truncate_to], label[:truncate_to]


    # https://stackoverflow.com/a/69877713
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        accuracy_sum = 0
        precision_t_sum = 0
        recall_t_sum = 0
        f1_t_sum = 0
        accuracy_t_sum = 0
        
        for i in range(len(predictions)):
            pred = predictions[i]
            label = labels[i]
            
            pred, label = self.truncate_padding(pred, label, Settings.TRANSFORMER_TRG_PAD_IDX)

            current_accuracy = self.accuracy_metric.compute(predictions=pred, references=label)
        
            # _t: abbreviation for _token
            fp_t, tp_t, fn_t, tn_t = Utils.get_fp_tp_fn_tn_token_based(label,
                                                                       pred,
                                                                       len(self.tf_tokenizer_pred.word_index) - 3,
                                                                       pred_empty_token=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                                       true_empty_token=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()])
            precision_score_t, recall_score_t, accuracy_score_t = Utils.precision(tp_t, fp_t), Utils.recall(tp_t, fn_t), \
                                                                Utils.accuracy(fp_t, tp_t, fn_t, tn_t)
            f1_score_t = Utils.f1(precision_score_t, recall_score_t)

            accuracy_sum += current_accuracy['accuracy']
            precision_t_sum += precision_score_t
            recall_t_sum += recall_score_t
            f1_t_sum += f1_score_t
            accuracy_t_sum += accuracy_score_t

        np.random.shuffle(self._train_set)
        np.random.shuffle(self._validation_set)
        train_batches = list(GPT2DatasetUtils.generate_torch_dataset_compatible_dataset_iterator(self._train_set[:250], self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE, self.max_length))
        validation_batches = list(GPT2DatasetUtils.generate_torch_dataset_compatible_dataset_iterator(self._validation_set[:250], self.tokenizer, self.tf_tokenizer_fit_on_dataset, self.BATCH_SIZE, self.max_length))

        UniversalAccess.output.write("Running inference on training set...")

        training_average_metrics, _ = self.gpt2_lmhead_utils.run_prediction(batches=train_batches,
                                                encoder=self.prot_t5_model.to(self.args.device),
                                                model=self.model_for_inference,
                                                caller=None,
                                                pred_SOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                pred_EOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                pred_EMPTY_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                pred_reverse_word_index=self.tf_tokenizer_pred_reverse_word_index,
                                                true_SOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                                true_EOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                                true_EMPTY_token_id=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                                true_reverse_word_index=self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                                compute_metrics=True,
                                                prefix="training_inference_")
        
        UniversalAccess.output.write("Running inference on validation set...")
        
        validation_average_metrics, _ = self.gpt2_lmhead_utils.run_prediction(batches=validation_batches,
                                        encoder=self.prot_t5_model.to(self.args.device),
                                        model=self.model_for_inference,
                                        caller=None,
                                        pred_SOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                        pred_EOS_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                        pred_EMPTY_token_id=self.tf_tokenizer_pred.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                        pred_reverse_word_index=self.tf_tokenizer_pred_reverse_word_index,
                                        true_SOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_SOS_TOKEN],
                                        true_EOS_token_id=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EOS_TOKEN],
                                        true_EMPTY_token_id=self.tf_tokenizer_fit_on_dataset.word_index[Settings.TRANSFORMER_EMPTY_TOKEN.lower()],
                                        true_reverse_word_index=self.tf_tokenizer_fit_on_dataset_reverse_word_index,
                                        compute_metrics=True,
                                        prefix="validation_inference_")

        return {'accuracy': accuracy_sum/len(predictions),
                'token_based_precision': precision_t_sum/len(predictions),
                'token_based_recall': recall_t_sum/len(predictions),
                'token_based_f1': f1_t_sum/len(predictions),
                'token_based_accuracy': accuracy_t_sum/len(predictions), **training_average_metrics, **validation_average_metrics}
    
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
    
    def evaluate_merged_mode_prediction(self,
                                        classification_predictions,
                                        transformer_predictions,
                                        classification_model_go_term_to_index,
                                        classification_model_reverse_go_term_to_index,
                                        transformer_model_go_term_to_index,
                                        transformer_model_reverse_go_term_to_index):
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        for idx in range(len(classification_predictions)):
            protein_sequence = self.dataset[idx][0].replace(" ", "")
            
            classification_prediction = classification_predictions[idx]
            transformer_prediction = transformer_predictions[idx]
            
            merged_prediction = set()
            
            for predicted_token in classification_prediction:
                merged_prediction.add(classification_model_reverse_go_term_to_index[predicted_token].upper())
            
            for predicted_token in transformer_prediction:
                if predicted_token != Settings.TRANSFORMER_EMPTY_TOKEN:
                    merged_prediction.add(transformer_model_reverse_go_term_to_index[predicted_token].upper())
            
            merged_prediction = list(merged_prediction)

            groundtruth_tokens = self.dataset[idx][1].split()[1:-1] # exclude <sos> and <eos> tokens at the beginning and the end
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
    
    def evaluate_classification_head_predictions(self,
                                                 classification_predictions,
                                                 classification_model_go_term_to_index,
                                                 classification_model_reverse_go_term_to_index):
        metrics = {"precision": 0, "recall": 0, "f1": 0}
        for idx in range(len(classification_predictions)):
            protein_sequence = self.dataset[idx][0].replace(" ", "")
            
            classification_prediction = classification_predictions[idx]
            
            merged_prediction = set()
            
            for predicted_token in classification_prediction:
                merged_prediction.add(classification_model_reverse_go_term_to_index[predicted_token].upper())

            merged_prediction = list(merged_prediction)

            groundtruth_tokens = self.dataset[idx][1].split()[1:-1] # exclude <sos> and <eos> tokens at the beginning and the end
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
                                                      self.model.get_config().go_term_to_index.get(Settings.TRANSFORMER_EMPTY_TOKEN.lower(), None))
            )
        )
        
        return string_io.getvalue()
