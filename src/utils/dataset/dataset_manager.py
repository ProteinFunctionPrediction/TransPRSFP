import numpy as np

from utils.dataset.dataset_utils import DatasetUtils
from universal.settings.settings import Settings
from universal.access.universal_access import UniversalAccess
from tensorflow.keras.preprocessing.text import Tokenizer

class DatasetManager:
    def __init__(self,
                 dataset,
                 cluster_mapping=None,
                 prot_seq_to_id_index=None,
                 equivalent_prot_id_list=None,
                 equivalent_prot_id_index_map=None):
        self.__dataset = dataset
        self.__cluster_mapping = cluster_mapping
        self.__prot_seq_to_id_index = prot_seq_to_id_index
        self.__equivalent_prot_id_list = equivalent_prot_id_list
        self.__equivalent_prot_id_index_map = equivalent_prot_id_index_map

        # The unwrapped version of the augmented dataset 
        # (Applicaple if and only if the dataset is augmented. Otherwise, it would simply be the same as the dataset itself)
        # The word "augmented" in the source code is used as a shortcut to mean that the dataset is a numpy array of such instances:
        # [protein_sequence, [go_term_sequences_str1, go_term_sequences_str2, ...]]
        # The unwrapping here means to convert the format above to the normal dataset format
        self.__dataset_unwrapped = dataset
        self.__is_dataset_augmented = DatasetUtils.is_dataset_augmented(dataset)
        if self.__is_dataset_augmented:
            self.__dataset_unwrapped = DatasetUtils.unwrap_augmented_dataset(dataset)
        
        self.__tf_tokenizer = self.__prepare_tf_tokenizer()
        self.__go_term_count = self.__compute_go_term_count()

        self.__training_set = None
        self.__validation_set = None
        
        self.__training_prot_ids = None
        self.__validation_prot_ids = None
        
        self.__training_cluster_stats = None
        self.__validation_cluster_stats = None
        
        self.__close_validation_set = None
        self.__distant_validation_set = None
        
        self.__training_based_cluster_split_stat = None
        self.__validation_based_cluster_split_stat = None
        
                
    def __prepare_tf_tokenizer(self):
        tf_tokenizer = Tokenizer(oov_token=Settings.TRANSFORMER_OOV_TOKEN, filters='')
        tf_tokenizer.fit_on_texts(self.__dataset_unwrapped[:, 1])
        return tf_tokenizer
    
    def __compute_go_term_count(self):
        return len(self.__tf_tokenizer.word_index) - 3
    
    def __split_augmented_dataset_to_train_val(self, val_count):
        dataset = self.__dataset
        val = []
        counts = DatasetUtils.count_go_term_datapoints(dataset, augmented=True)
        remove_proteins = set()
        count = 0
        for i in dataset:
            prot_seq, go_term_sequences = i
            go_term_counts = DatasetUtils.count_go_term_datapoints_in_single_augmented_sample(go_term_sequences)
            skip = False
            for go_term, go_term_count in go_term_counts.items():
                if counts[go_term] <= go_term_count:
                    skip = True
                    break
            if skip:
                continue

            for go_term, go_term_count in go_term_counts.items():
                counts[go_term] -= go_term_count
            val.append(i)
            remove_proteins.add(prot_seq)
            count += 1

            if count == val_count:
                break

        train = []
        for i in dataset:
            prot_seq, go_terms = i
            if prot_seq not in remove_proteins:
                train.append(i)

        self.__training_set = np.asarray(train)
        self.__validation_set = np.asarray(val)
    
    def __split_normal_dataset_to_train_val(self, val_count):
        dataset = self.__dataset
        val = []
        counts = DatasetUtils.count_go_term_datapoints(dataset)
        remove_proteins = set()
        count = 0
        for i in dataset:
            prot_seq, go_terms = i
            s = set(go_terms.split()[1:-1])
            skip = False
            for go_term in s:
                if counts[go_term] <= 1:
                    skip = True
                    break
            if skip:
                continue

            for go_term in s:
                counts[go_term] -= 1
            val.append(i)
            remove_proteins.add(prot_seq)
            count += 1

            if count == val_count:
                break

        train = []
        for i in dataset:
            prot_seq, go_terms = i
            if prot_seq not in remove_proteins:
                train.append(i)

        self.__training_set = np.asarray(train)
        self.__validation_set = np.asarray(val)
    
    def split_train_val(self, val_count):
        assert self.__training_set is None and self.__validation_set is None
        if self.__is_dataset_augmented:
            self.__split_augmented_dataset_to_train_val(val_count)
        else:
            self.__split_normal_dataset_to_train_val(val_count)
        
        UniversalAccess.output.write(f"Number of proteins in the training set: {len(self.__training_set)}")
        UniversalAccess.output.write(f"Number of proteins in the validation set: {len(self.__validation_set)}")
        
        if self.__cluster_mapping is not None and self.__prot_seq_to_id_index is not None:
            self.__training_prot_ids = self.__extract_prot_ids_from_dataset(self.__training_set)
            self.__validation_prot_ids = self.__extract_prot_ids_from_dataset(self.__validation_set)
            
            self.__training_cluster_stats = self.__get_cluster_stats(self.__training_prot_ids)
            self.__validation_cluster_stats = self.__get_cluster_stats(self.__validation_prot_ids)

            self.__training_based_cluster_split_stat = self.__compute_cluster_split_stats(self.__training_cluster_stats, self.__validation_cluster_stats)
            self.__validation_based_cluster_split_stat = self.__compute_cluster_split_stats(self.__validation_cluster_stats, self.__training_cluster_stats)
            
            self.__report_cluster_stats()
            self.__split_validation_set_based_on_cluster()
    
    def __report_cluster_stats(self):     
        UniversalAccess.output.write(f"There are {self.__training_cluster_stats['unknown']} proteins whose cluster is not known in the training set")
        UniversalAccess.output.write(f"There are {self.__validation_cluster_stats['unknown']} proteins whose cluster is not known in the valdation set")
        
        UniversalAccess.output.write(f"There are {self.__training_based_cluster_split_stat} proteins in the training set for which there is at least one protein in the same cluster in the validation set")
        UniversalAccess.output.write(f"There are {self.__validation_based_cluster_split_stat} proteins in the validation set for which there is at least one protein in the same cluster in the training set")

        
    def __split_validation_set_based_on_cluster(self):
        assert self.__close_validation_set is None and self.__distant_validation_set is None
        close_validation_set = []
        distant_validation_set = []
        
        unknown_count = 0
        
        for prot_seq, label in self.__validation_set:
            cluster_ids = self.__get_cluster_ids_of_prot_seq(prot_seq)
            if cluster_ids is None:
                unknown_count += 1
                continue

            distant = True
            for cluster_id in cluster_ids:
                if cluster_id in self.__training_cluster_stats:
                    distant = False
                    break

            if distant:
                distant_validation_set.append([prot_seq, label])
            else:
                close_validation_set.append([prot_seq, label])
        
        assert len(close_validation_set) == self.__validation_based_cluster_split_stat
        assert len(close_validation_set) + len(distant_validation_set) + unknown_count == len(self.__validation_set)
        
        self.__distant_validation_set = np.asarray(distant_validation_set, dtype=object)
        self.__close_validation_set = np.asarray(close_validation_set, dtype=object)
        
        
    
    def __get_cluster_ids_of_prot_seq(self, prot_seq):
        prot_id = self.__prot_seq_to_id_index.get(prot_seq, None)
        if prot_id is None:
            return None
        
        return self.__cluster_mapping.get(prot_id, None)
        
    
    def __compute_cluster_split_stats(self, cluster_stats1, cluster_stats2):
        result = 0
        for cluster in cluster_stats1:
            if cluster == "unknown":
                continue
            if cluster in cluster_stats2:
                result += cluster_stats1[cluster]
        return result
    
    def __extract_prot_ids_from_dataset(self, dataset):
        prot_ids = set()
        for prot_sequence, _ in dataset:
            prot_id = self.__prot_seq_to_id_index.get(prot_sequence, None)
            if prot_id is not None:
                prot_ids.add(prot_id)
        return prot_ids
        
    def __get_cluster_stats(self, prot_ids):
        stats = {"unknown": 0}
        for prot_id in prot_ids:
            cluster_ids = self.__cluster_mapping.get(prot_id, None)
            if cluster_ids is None and self.__equivalent_prot_id_list is not None and self.__equivalent_prot_id_index_map is not None:
                prot_id_syns = self.__get_prot_ids_synonyms(prot_id)
                for pid in prot_id_syns:
                    cluster_ids = self.__cluster_mapping.get(pid, None)
                    if cluster_ids is not None:
                        break
            
            if cluster_ids is None:
                stats["unknown"] += 1
                continue
            
            for cluster_id in cluster_ids:
                if cluster_id not in stats:
                    stats[cluster_id] = 0
                stats[cluster_id] += 1
        return stats
    
    def __get_prot_ids_synonyms(self, prot_id):
        idx = self.__equivalent_prot_id_index_map.get(prot_id, None)
        if idx is None:
            return set([prot_id])

        equivalency_set = equivalent_prot_id_list[idx]
        assert prot_id in equivalency_set
        
        return equivalency_set            

    def get_training_count(self, training_dataset_ratio):
        return DatasetUtils.get_training_count(training_dataset_ratio, len(self.__dataset_unwrapped))

    def get_datapoint_count(self):
        return len(self.__dataset_unwrapped)

    def get_go_term_count(self):
        return self.__go_term_count
    
    def get_tf_tokenizer(self):
        return self.__tf_tokenizer
    
    def get_dataset(self):
        return self.__dataset
    
    def get_unwrapped_dataset(self):
        return self.__dataset_unwrapped

    def is_dataset_augmented(self):
        return self.__is_dataset_augmented
    
    def shuffle(self):
        np.random.shuffle(self.__dataset)
        self.__dataset_unwrapped = self.__dataset
        if self.__is_dataset_augmented:
            self.__dataset_unwrapped = DatasetUtils.unwrap_augmented_dataset(self.__dataset)
    
    def is_distant_validation_set_split_available(self):
        return self.__distant_validation_set is not None
    
    def get_close_validation_set(self):
        assert self.__close_validation_set is not None
        return self.__close_validation_set
    
    def get_distant_validation_set(self):
        assert self.__distant_validation_set is not None
        return self.__distant_validation_set
    
    def get_training_set(self):
        return self.__training_set

    def get_validation_set(self):
        return self.__validation_set