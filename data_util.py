import enum
from datasets import dataset_dict, load_dataset,DatasetDict
from networkx.readwrite.json_graph import adjacency
from transformers import  DataCollatorForSeq2Seq
import nltk
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import stanza
#logger = logging.getLogger(__name__)
import os
import math

# own
from .CONSTANT import DISCOURSE_RELATIONS


def get_dataset(data_args,model_args):
    
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        
        datasets = load_dataset(extension, data_files=data_files)

    return datasets
    
def data_preprocessing(datasets,tokenizer,training_args,data_args,model_args):
    
    # if data_args.save_dataset_path is None or data_args.reprocess_data:
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    
    
    text_column = data_args.text_column
    summary_column = data_args.summary_column
    speaker_column = data_args.speaker_column
    unique_speaker_column = data_args.unique_speaker_column  # maybe we should dynamically add new column?

    padding = "max_length" if data_args.pad_to_max_length else False
       
    def preprocess_function(examples):
        ## examples:{str:[List[List]]}
        source = examples[text_column] # List[List[str]]
        targets = examples[summary_column] #  List[str]
        speakers = examples[speaker_column] # List[List[str]]
        unique_speakers = examples[unique_speaker_column] #List[List[str]]
        ids = examples['id']


        bs = len(source)
        if model_args.model_type == 'transformer':

            model_inputs = {'gt_input_ids': [],
                            'gt_attention_mask': [],
                            'id':[]}
            if "SDG_adj" in model_args.feature_types:
                model_inputs['SDG_adj'] = []


            inputs = []
            for batch_idx in range(bs):
                utts = source[batch_idx] #List[str]
                if discourse_relations:
                    discourse_relations_per_instance = discourse_relations[batch_idx] #List[List[int,str,int]]
                if keywords:
                    keywords_per_instance = keywords[batch_idx]

                if len(utts) > data_args.max_utt_num:
                    utts = utts[:data_args.max_utt_num]
                    unique_speakers[batch_idx] = unique_speakers[batch_idx][:data_args.max_utt_num]
                    speakers[batch_idx] = speakers[batch_idx][:data_args.max_utt_num]

                # print('utts:',utts)

                tokenized_utts = tokenizer(utts, max_length=data_args.max_seq_len_per_utt, padding=padding, truncation=True)
                model_inputs['gt_input_ids'].append(tokenized_utts['input_ids'])
                model_inputs['gt_attention_mask'].append(tokenized_utts['attention_mask'])
                model_inputs['id'].append(ids[batch_idx])
                if "SDG_adj" in model_args.feature_types:
                    model_inputs['SDG_adj'].append(get_distance_adj(len(utts)))
                

                if '<sep>' not in tokenizer.additional_special_tokens:
                    special_tokens_dict = {"additional_special_tokens":["<sep>"]}
                    tokenizer.add_special_tokens(special_tokens_dict)

                inputs_str = ""
                for utt_idx in range(min(data_args.max_utt_num,len(source[batch_idx]))):
                    inputs_str += speakers[batch_idx][utt_idx]
                    inputs_str += ': '
                    inputs_str += source[batch_idx][utt_idx]
                    inputs_str += ' <sep> '
                inputs.append(inputs_str)
            baseline_input = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
            model_inputs['input_ids'] = baseline_input['input_ids']
            model_inputs['attention_mask'] = baseline_input['attention_mask']

        elif model_args.model_type == 'baseline':
            if '<sep>' not in tokenizer.additional_special_tokens:
                special_tokens_dict = {"additional_special_tokens":["<sep>"]}
                tokenizer.add_special_tokens(special_tokens_dict)

            inputs = []
            for batch_idx in range(bs):
                inputs_str = ""
                for utt_idx in range(len(source[batch_idx])):

                    inputs_str += speakers[batch_idx][utt_idx]
                    inputs_str += ": "
                    inputs_str += source[batch_idx][utt_idx]
                    inputs_str += ' <sep> '
                inputs_str = inputs_str[:-7] # delete last <sep>
                inputs.append(inputs_str)
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True,add_special_tokens=False)
            for k,v in labels.items():
                if k == 'input_ids':
                    labels[k] = [x+[tokenizer.eos_token_id] for x in v]
                elif k == 'attention_mask':
                    labels[k] = [x+[1] for x in v]

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    output_datasets = [None,None,None]
    ## dataset mappping
    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[0] = train_dataset

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[1] = eval_dataset

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[2]=predict_dataset
    return output_datasets

def get_SDG_adj(num_utt):
    
    ret = []
    for idx in range(num_utt):
        start = 0-idx
        row = []
        while len(row) < num_utt:
            row.append(start)
            start += 1
        ret.append(row)
    return ret


#.......
