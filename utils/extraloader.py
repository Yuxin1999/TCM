import torch 
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

class ExtraLoader:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.max_length = args.max_length

        self.standard_path = args.standard_path
        self.therapy_word_path = args.therapy_word_path
        self.therapy_kg_embedding_path = args.therapy_kg_embedding_path
        self.symptom_kg_embedding_path = args.symptom_kg_embedding_path
        self.relation_embedding_path = args.relation_embedding_path
        
    def load_standard_encoding(self):
        with open(self.standard_path) as f:
            lines = f.readlines()
        
        standard_encodings = []
        for line in lines:
            words = line.strip().split()
            encoding = self.tokenizer(words, add_special_tokens=True, truncation=True, padding='max_length',
                                        max_length=self.max_length, return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            standard_encodings.append({'input_ids': input_ids, 'attention_mask': attention_mask})

        return standard_encodings

    def load_therapy_encodings(self):
        with open(self.therapy_word_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        
        therapy_encodings = []
        for line in lines:
            word = line.strip()
            encoding = self.tokenizer(word, add_special_tokens=True, truncation=True, padding='max_length',
                                        max_length=self.max_length, return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            therapy_encodings.append({'input_ids': input_ids, 'attention_mask': attention_mask})
        
        return therapy_encodings
    
    def load_therapy_kg_embedding(self):
        numpy = np.load(self.therapy_kg_embedding_path)
        therapy_kg_embedding = torch.from_numpy(numpy)
        return therapy_kg_embedding

    def load_symptom_kg_embedding(self):
        numpy = np.load(self.symptom_kg_embedding_path)
        symptom_kg_embedding = torch.from_numpy(numpy)
        return symptom_kg_embedding

    def load_relation_embedding(self):
        numpy = np.load(self.relation_embedding_path)
        relation_embedding = torch.from_numpy(numpy)
        return relation_embedding