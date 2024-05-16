import json
import numpy as np
import csv 
import torch 
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

class DataModule:
    def __init__(self, args):
        self.input_path = args.input_path
        self.therapy_word_path = args.therapy_word_path
        self.standard_path = args.standard_path

        self.therapy_word_path = args.therapy_word_path

        self.random_seed = args.random_seed
        self.max_length = args.max_length
        self.batch_size = args.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        self.train_dataset = None
        self.test_dataset = None
        self.dev_dataset = None

        self.test_size = args.test_size 

    def setup(self, stage=None):
        data = json.load(open(self.input_path,'r',encoding='utf-8'))
        symptoms = []
        no_symptoms = []
        therapys = []

        for entry in data:
            symptom_data = entry["symptom"]
            no_symptom_data = entry["no_symptom"]
            therapy_data = entry["therapy"]
            
            symptoms.append(symptom_data)
            no_symptoms.append(no_symptom_data)
            therapys.append(therapy_data)
        
        train_symptoms, val_symptoms, train_no_symptoms, val_no_symptoms, train_therapys, val_therapys = train_test_split(
            symptoms, no_symptoms, therapys, test_size=self.test_size, random_state=self.random_seed
        )

        dev_symptoms, test_symptoms, dev_no_symptoms, test_no_symptoms, dev_therapys, test_therapys = train_test_split(
            val_symptoms, val_no_symptoms, val_therapys, test_size=0.5, random_state=self.random_seed
        )

        therapy_list = []
        with open(self.therapy_word_path,'r',encoding='utf-8') as f:
            for line in f:
                therapy_list.append(line.strip())
        
        symptom_list = []
        with open(self.standard_path,'r',encoding='utf-8') as f:
            for line in f:
                symptom_list.append(line.strip())

        self.train_dataset = MyDataset(train_symptoms, train_no_symptoms, train_therapys, self.tokenizer, self.max_length, therapy_list, symptom_list)
        self.test_dataset = MyDataset(test_symptoms, test_no_symptoms, test_therapys, self.tokenizer, self.max_length, therapy_list, symptom_list)
        self.dev_dataset = MyDataset(dev_symptoms, dev_no_symptoms, dev_therapys, self.tokenizer, self.max_length, therapy_list, symptom_list)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, drop_last=True)
    
    def dev_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)

    def print_size(self):
        print("Train Dataset: ", len(self.train_dataset))
        print("Dev Dataset: ", len(self.dev_dataset))
        print("Test Dataset:", len(self.test_dataset))


class MyDataset(Dataset):
    def  __init__(self, symptoms, no_symptoms, therapy, tokenizer, max_length, therapy_list, symptom_list) -> None:
        super().__init__()
        self.symptoms = symptoms
        self.no_symptoms = no_symptoms
        self.therapy = therapy 

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.therapy_list = therapy_list

        self.symptom_list = symptom_list
    
    def __len__(self):
        return len(self.therapy)
    
    def __getitem__(self, index):
        symptoms = self.symptoms[index]
        no_symptoms = self.no_symptoms[index]
        therapy = self.therapy[index]

        symptoms_input_ids = []
        symptoms_attention_masks = []
        symptoms_ids = []

        no_symptoms_input_ids = []
        no_symptoms_attention_masks = []
        no_symptoms_ids = []

        for symptom in symptoms:
            symptom_id = self.symptom_list.index(symptom)
            symptoms_ids.append(symptom_id)

            encoding = self.tokenizer.encode_plus(
            symptom,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
            
            symptom_input_id = encoding['input_ids'].squeeze()
            symptom_attention_mask = encoding['attention_mask'].squeeze()

            symptoms_input_ids.append(symptom_input_id)
            symptoms_attention_masks.append(symptom_attention_mask)
        
        for no_symptom in no_symptoms:
            symptom_id = self.symptom_list.index(no_symptom)
            no_symptoms_ids.append(symptom_id)

            encoding = self.tokenizer.encode_plus(
            no_symptom,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
            
            no_symptom_input_id = encoding['input_ids'].squeeze()
            no_symptom_attention_mask = encoding['attention_mask'].squeeze()

            no_symptoms_input_ids.append(no_symptom_input_id)
            no_symptoms_attention_masks.append(no_symptom_attention_mask)

        
        therapy_encoding = [0]*len(self.therapy_list)
        index = self.therapy_list.index(therapy)
        therapy_encoding[index] = 1

        return {
            'input_ids': symptoms_input_ids,
            'attention_mask': symptoms_attention_masks,
            'symptom_ids': symptoms_ids,
            "no_input_ids": no_symptoms_input_ids,
            "no_attention_mask": no_symptoms_attention_masks,
            'no_symptom_ids': no_symptoms_ids,
            'therapy': therapy_encoding
        }


