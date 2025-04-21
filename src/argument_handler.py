import argparse
import os
from universal.settings.settings import Settings

class ArgumentHandler:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        group = self.parser.add_argument_group('arguments')
        group.add_argument('-i', '--inference', action='store_true', help='If it is given, the model is set to run in inference mode')
        group.add_argument('-t', '--train', action='store_true', help='If it is given, the model will be trained')
        group.add_argument('-d', '--device', help='the device on which the model is to run', default='cpu')
        
        group.add_argument('-dp', '--dataset', help='path to dataset (must be a pickle-saved binary file)', required=True)
        group.add_argument('-mt', '--model-type', choices=[Settings.TRANSFORMER_MODEL_TYPE, 
                                                           Settings.CLASSIFICATION_HEAD_MODEL_TYPE,
                                                           Settings.GPT2_MODEL_TYPE,
                                                           Settings.MERGED_MODEL_TYPE],
                           required=True)
        group.add_argument('-tmt', '--transformer-model-type', choices=[Settings.TRANSFORMER_MODEL_TYPE, Settings.GPT2_MODEL_TYPE], help='required only if the merged model is used')
        group.add_argument('-mp', '--model-path', help='path to the directory where the model is saved')
        group.add_argument('-t5', '--prot-t5-model-path', help='path to the directory where ProtT5 model is stored', required=True)
        group.add_argument('-th', '--threshold', type=float, help='threshold for classification head model.\nThe less the threshold, the more the number of false positives, but the less the number of false negatives.\nThe more the threshold, the less the number of false positives, but the more the number of false negatives.')
        group.add_argument('-bs', '--batch-size', type=int, help='batch size')
        group.add_argument('-e', '--epoch', type=int, help='number of epochs for training')
        group.add_argument('-lr', '--learning-rate', type=float, help='learning rate for training')
        group.add_argument('-msd', '--model-save-dir', type=str, help="the folder path to save the model(s) (must be non-existent)")
        group.add_argument('-spe', '--save-per-epoch', type=int, help='if set to a positive value, each x epochs, the model is saved to\nthe model save dir specified in the arguments', default=0)
        group.add_argument('-ml', '--max-length', type=int, help='the maximum length for a protein sequence. If not set, automatically inferred from the dataset', default=0)
        group.add_argument('-tdr', '--training-dataset-ratio', type=float, help='the ratio of the number of samples in training data to the number of samples in all dataset (must be between 0 and 1, 0 and 1 excluded)')
        group.add_argument('-tbld', '--tensorboard-log-dir', type=str, help='the folder where the tensorboard logs are to be saved (must be non-existent)')
        group.add_argument('-chmp', '--classification-head-model-path', type=str, help='path to the classification head model (only used in merged model type)')
        group.add_argument('-tmp', '--transformer-model-path', type=str, help='path to the transformer model (only used in merged model type)')
        group.add_argument('-rn', '--run-name', type=str, help='the name identifying this run', default='test')
        group.add_argument('-gti', '--go-term-index', type=str, help='the file path to save GO term index (must be non-existent)')
        group.add_argument('-lgti', '--load-go-term-index', type=str)
        group.add_argument('-s', '--random-seed', type=int, help='random seed', required=False, default=42)
        group.add_argument('-ns', '--no-random-seed', action='store_true', help='if specified, no constant will be set as random seed')
        group.add_argument('-cfp', '--continue-from-checkpoint', type=str, required=False, help='path to a checkpoint of the model to be trained')
        group.add_argument('-gtmfp', '--go-term-metrics-filepath-prefix', type=str, required=False, help='the prefix of filenames to save GO term-based metrics.\nIf not specified, a random string will be generated. This option can only be used in inference mode', default='')
        group.add_argument('-cm', '--compute-metrics', action='store_true', help='If it is given, task-specific metrics will be computed and reported. This can be used only in inference mode.\nIf the provided dataset does not properly include the corresponding true labels, the execution will fail and terminate at an early stage')
        group.add_argument('-stst', '--save-training-set-to', type=str, required=False, help='the path to save the training set. The specified path must be non-existent. This option can only be used in training mode')
        group.add_argument('-stest', '--save-test-set-to', type=str, required=False, help='the path to save the test set. The specified path must be non-existent. This option can only be used in training mode')
        
        self.args = self.parser.parse_args()
        
        self._validate()
    
    def _validate(self):
        if self.args.load_go_term_index is not None and not os.path.exists(self.args.load_go_term_index):
            self._show_help_and_raise_error(f"The specified GO term index file path does not exist!: {self.args.load_go_term_index}")
        
        if self.args.go_term_index is not None and os.path.exists(self.args.go_term_index):
            self._show_help_and_raise_error("The GO term index file path must be non-existent")

        if not self.args.inference and not self.args.train:
            self._show_help_and_raise_error("Neither inference mode nor training mode is set!")
        
        if self.args.model_type == Settings.MERGED_MODEL_TYPE:
            if not self.args.inference:
                self._show_help_and_raise_error("The merged model type can be run only in inference mode, but inference mode is not set!")
            
            if self.args.transformer_model_type is None:
                self._show_help_and_raise_error("The transformer model type has to be specified when the merged model is used!")
            
            if self.args.classification_head_model_path is None:
                self._show_help_and_raise_error("Classification head model path must be specified when running in merged model mode!")
            if not os.path.exists(self.args.classification_head_model_path):
                self._show_help_and_raise_error(f"No such directory!: {self.args.classification_head_model_path}")
                
            if self.args.transformer_model_path is None:
                self._show_help_and_raise_error("Transformer model path must be specified when running in merged model mode!")
            if not os.path.exists(self.args.transformer_model_path):
                self._show_help_and_raise_error(f"No such directory!: {self.args.transformer_model_path}")
                
            if self.args.model_path is not None:
                self._show_help_and_raise_error("Model path argument cannot be used when running in merged model mode!")
        else:
            if self.args.classification_head_model_path is not None:
                self._show_help_and_raise_error("Classification head model path can be specified only when running in merged model mode!")
            if self.args.transformer_model_path is not None:
                self._show_help_and_raise_error("Transformer model path can be specified only when running in merged model mode!")
            
        
        if self.args.inference and self.args.train:
            self._show_help_and_raise_error("Train and inference modes cannot be set at the same time!")
        
        if not os.path.isfile(self.args.dataset):
            self._show_help_and_raise_error(f"No such file!:{self.args.dataset}")
        
        
        
        if not os.path.isdir(self.args.prot_t5_model_path):
            self._show_help_and_raise_error(f"No such directory!: {self.args.prot_t5_model_path}")
        
        if self.args.max_length < 0:
            self._show_help_and_raise_error(f"Maximum length must not be negative!")
        
        if self.args.inference:
            if self.args.model_type != Settings.MERGED_MODEL_TYPE:
                if self.args.model_path is None:
                    self._show_help_and_raise_error("Model path must be specified when running in inference mode!")

                if not os.path.isdir(self.args.model_path):
                    self._show_help_and_raise_error(f"No such directory!: {self.args.model_path}")
            
            if (self.args.model_type == Settings.CLASSIFICATION_HEAD_MODEL_TYPE or self.args.model_type == Settings.MERGED_MODEL_TYPE) and not self.args.threshold:
                self._show_help_and_raise_error("You have to specify the threshold when using the classification head model in inference mode!")

            if self.args.save_training_set_to is not None:
                self._show_help_and_raise_error("--save-training-set-to parameter cannot be used in inference mode!")

            if self.args.save_test_set_to is not None:
                self._show_help_and_raise_error("--save-test-set-to parameter cannot be used in inference mode!")
            
        
        if self.args.train:
            if self.args.batch_size is None:
                self._show_help_and_raise_error("You have to specify the batch size in training mode!")
            elif self.args.batch_size <= 0:
                self._show_help_and_raise_error("Batch size must be a positive integer!")

            if self.args.epoch is None:
                self._show_help_and_raise_error("You have to specify number of epochs in training mode!")
            elif self.args.epoch <= 0:
                self._show_help_and_raise_error("Epoch must be a positive integer!")
                
            if self.args.learning_rate is None:
                self._show_help_and_raise_error("You have to specify the learning rate in training mode!")
            elif self.args.learning_rate <= 0:
                self._show_help_and_raise_error("Learning rate must be positive!")
                
            if self.args.model_save_dir is None:
                self._show_help_and_raise_error("You have to specify the model save dir in training mode!")
            elif os.path.exists(self.args.model_save_dir):
                self._show_help_and_raise_error("The model save dir must not exist!")
                
            if self.args.save_per_epoch < 0:
                self.args.save_per_epoch = 0
                
            if self.args.save_per_epoch >= self.args.epoch:
                self._show_help_and_raise_error("Save per epoch cannot be greater than or equal to the epoch number!")
            
            if self.args.training_dataset_ratio is None:
                self._show_help_and_raise_error("You have to specify the training dataset ratio in training mode!")
            elif self.args.training_dataset_ratio < 0 or self.args.training_dataset_ratio > 1:
                self._show_help_and_raise_error("Training dataset ratio must be between 0 and 1 (0 and 1 excluded)!")
            elif self.args.training_dataset_ratio == 0 or self.args.training_dataset_ratio == 1:
                self._show_help_and_raise_error("Training dataset ratio cannot be 0 or 1!")
            
            if self.args.model_type == Settings.TRANSFORMER_MODEL_TYPE:
                if self.args.tensorboard_log_dir is None:
                    self._show_help_and_raise_error("You have to specify the tensorboard log dir while running transformer model in training mode!")
                
                if os.path.exists(self.args.tensorboard_log_dir):
                    self._show_help_and_raise_error("The speficied tensorboard log dir must not exist!")

            if self.args.save_training_set_to is not None:
                if os.path.exists(self.args.save_training_set_to):
                    self._show_help_and_raise_error(f"The path specified as --save-training-set-to parameter must not exist!: {self.args.save_training_set_to}")
                
                if not os.path.isdir(os.path.split(self.args.save_training_set_to)[0]):
                    self._show_help_and_raise_error(f"Not a directory: {os.path.split(self.args.save_training_set_to)[0]}")

            if self.args.save_test_set_to is not None:
                if os.path.exists(self.args.save_test_set_to):
                    self._show_help_and_raise_error(f"The path specified as --save-test-set-to parameter must not exist!: {self.args.save_test_set_to}")
                
                if not os.path.isdir(os.path.split(self.args.save_test_set_to)[0]):
                    self._show_help_and_raise_error(f"Not a directory: {os.path.split(self.args.save_test_set_to)[0]}")
            
            if self.args.go_term_metrics_filepath_prefix != '':
                self._show_help_and_raise_error("--go-term-metrics-filepath-prefix option cannot be used in training mode!")
            
            if self.args.compute_metrics:
                self._show_help_and_raise_error("--compute-metrics option cannot be used in training mode!")

    def _show_help_and_raise_error(self, message: str, error_type=RuntimeError) -> None:
        self.parser.print_help()
        print("\nYou are seeing this message since the validation of given arguments failed due to the following reason:")
        raise error_type(message)
    
    def get_args(self):
        return self.args