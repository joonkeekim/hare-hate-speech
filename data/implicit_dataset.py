from torch.utils.data import Dataset
import pandas as pd
import json


# dataset for baseline c model
class ImplicitDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        
        post = example['post']
        hate_class = example['class']

        query = f'Determine whether the following post is hate speech. You have to choose one of the options.\nPost: {post}\nOptions: \n(A) Hate\n(B) Not hate\nAnswer:\n'
        if hate_class == "not_hate":
            target = "(B) Not hate."
        elif hate_class == "implicit_hate" or hate_class == "explicit_hate":
            target = "(A) Hate."
        
        return {
            'query': query,
            'target': target,
        }
        
# dataset for HARE(including c+t+i)
class ImplicitReasoningDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        query = example['ft_query']
        target = example['ft_target']

        return {
            'query': query,
            'target': target,
        }
    
# dataset for cross evaluation 
class IH2HateDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())#[:len(data)//100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        post = example['post']
        label = example['label']
        query = f'Determine whether the following post is hate speech. You have to choose one of the options.\nPost: {post}\nOptions: \n(A) Hate\n(B) Not hate\nAnswer:\n'
        if label == "nothate" or label =="normal":
            target = "(B) Not hate."
        else:
            target = "(A) Hate."

        return {
            'query': query,
            'target': target,
        }

class ImplicitCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, sequences):
        quries = [sequence['query'] for sequence in sequences]
        targets = [sequence['target'] for sequence in sequences]
        
        query_encoding_dict = self.tokenizer(quries, return_tensors='pt', padding=True, max_length=512)
        query_input_ids = query_encoding_dict['input_ids']
        query_masks = query_encoding_dict['attention_mask']
        
        targets_encoding_dict = self.tokenizer(targets, return_tensors='pt', padding=True, max_length=512)
        targets_input_ids = targets_encoding_dict['input_ids']
        targets_masks = targets_encoding_dict['attention_mask']
        
        return {
            "query_inputs": query_input_ids, 
            "query_masks":query_masks, 
            "target_inputs": targets_input_ids, 
            "target_masks": targets_masks
            }