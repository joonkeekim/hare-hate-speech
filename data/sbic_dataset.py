from torch.utils.data import Dataset
import json


# dataset for baseline c model
class SBICDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        query = f'Determine whether the following post is offensive or not. You must choose one of the options.\nPost: {example["post"]}\nOptions: \n(A) Offensive\n(B) Not offensive\nAnswer:\n'
        label = example['label_off']
        if label == "not offensive":
            target = "The answer is : (B) Not offensive"
        else:
            target = "The answer is : (A) Offensive"

        return {
            'query': query,
            'target': target,
        }
    
# dataset for cross evaluation 
class SBIC2HateDataset(Dataset):
    def __init__(self, data_path):
        # self.data = pd.read_csv(data_path)
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())#[:len(data)//100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        query = f'Determine whether the following post is offensive or not. You must choose one of the options.\nPost: {example["post"]}\nOptions: \n(A) Offensive\n(B) Not offensive\nAnswer:\n'
        label = example['label']
        if label == "nothate" or label =="normal":
            target = "The answer is : (B) Not offensive"
        else:
            target = "The answer is : (A) Offensive"

        return {
            'query': query,
            'target': target,
        }


    
# dataset for HARE
class SBICReasoningDataset(Dataset):
    def __init__(self, data_path, train=False):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())
        self.is_train = train
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        
        if self.is_train:
            label = example['label_off']
            if ('(A)' in example['response'] and example['label_off'] == 'offensive') or \
               ('(B)' in example['response'] and example['label_off'] == 'not offensive'):  
                
                if label == "not offensive":
                    target = "The answer is : (B) Not offensive"
                else:
                    target = "The answer is : (A) Offensive"
                target += example['ft_target']
            else:
                if label == "not offensive":
                    target = "The answer is : (B) Not offensive"
                else:
                    target = "The answer is : (A) Offensive"

        else:
            label = example['label_off']
            if label == "not offensive":
                target = " (B) Not offensive"
            else:
                target = " (A) Offensive"

        return {
            'query': example['ft_query'],
            'target': target,
        }

# dataset for c+t+i option
class SBICgiven(Dataset):
    def __init__(self, data_path):
        # self.data = pd.read_csv(data_path)
        # import pdb; pdb.set_trace()
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.data = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        query = f'Determine whether the following post is offensive or not. You must choose one of the options.\nPost: {example["post"]}\nOptions: \n(A) Offensive\n(B) Not offensive\nAnswer:\n'
        label = example['label_off']
        if label == "not offensive":
            target = "The answer is : (B) Not offensive"
        else:
            if ('target' in example and 'implied_statement' in example) and (example['target'] != '[]' and example['implied_statement'] != '[]'):
                target = f"The answer is : (A) Offensive. The post targets {eval(example['target'])[0]} and implies {eval(example['implied_statement'])[0]}."
            else:
                target = "The answer is : (A) Offensive."

        return {
            'query': query,
            'target': target,
        }

class SBICCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, sequences):
        quries = [sequence['query'] for sequence in sequences]
        targets = [sequence['target'] for sequence in sequences]
        
        query_encoding_dict = self.tokenizer(quries, return_tensors='pt', padding=True,  max_length=512)
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