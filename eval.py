from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import json
from tqdm import tqdm
import warnings
import os
import pandas as pd
import numpy as np
import evaluate

# Ignore warnings
warnings.filterwarnings('ignore')

class Metrics():
    def __init__(self,tokenizer, output_dir, zeroshot) -> None:
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def save_preds(self, decoded_preds, decoded_trues):
        """
            save predictions
        """
        with open(os.path.join(self.output_dir, 'results.json'), "w") as file:
            # Iterate over the list and write each string to a new line in the file
            results = {}
            
            for i, (pred, true) in enumerate(zip(decoded_preds, decoded_trues)):
                results[i] = {"id": i, "pred":pred, "true":true}
            
            json.dump(results, file, indent=4)


    def decode_preds(self, preds):
        preds[preds == -100] = self.tokenizer.eos_token_id
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        return decoded_preds
    
    def decode_p(self, p):
        preds, labels = p.predictions, p.label_ids
        decoded_preds = self.decode_preds(preds)
        decoded_trues = self.decode_preds(labels)
        return (decoded_preds, decoded_trues)
        
    def compute_implicit_metrics(self, p):
        decoded_preds, decoded_trues = self.decode_p(p)
        self.save_preds(decoded_preds, decoded_trues)

        pred_labels, true_labels = [], []
        do_f1 = True
        for decoded_true, decoded_pred in zip(decoded_trues, decoded_preds):
            if decoded_pred.startswith("(A) Hate"):
                pred_label = "(A) Hate."
            elif decoded_pred.startswith("(B) Not hate"):
                pred_label = "(B) Not hate."
            else:
                do_f1 = False
                continue
                
            true_labels.append(decoded_true)
            pred_labels.append(pred_label)

        cls_score = accuracy_score(true_labels, pred_labels)
        if do_f1:
            f1 = f1_score(true_labels, pred_labels, average='binary',pos_label="(A) Hate.")
            pr_score = precision_score(true_labels, pred_labels, average="binary",pos_label="(A) Hate.")
            re_score = recall_score(true_labels, pred_labels, average="binary", pos_label="(A) Hate.")
        else:
            f1 = 0.0
            pr_score = 0.0
            re_score = 0.0
        
        return {"cls_acc": cls_score ,"cls_f1":f1, "cls_pr":pr_score, "cls_re":re_score}
    
    def compute_sbic_metrics(self, p):
        decoded_preds, decoded_trues = self.decode_p(p)
        self.save_preds(decoded_preds, decoded_trues)

        pred_labels, true_labels = [], []
        do_f1 = True
        for decoded_true, decoded_pred in zip(decoded_trues, decoded_preds):
            if "the answer is : (b) not offensive" in decoded_pred.lower():
                pred_label = 0
            elif "the answer is : (a) offensive" in decoded_pred.lower():
                pred_label = 1
            else:
                do_f1 = False
                continue

            if "(a)" in decoded_true.lower():
                true_label = 1
            elif "(b)" in decoded_true.lower():
                true_label = 0
            else:
                continue
                
            true_labels.append(true_label)
            pred_labels.append(pred_label)

        cls_score = accuracy_score(pred_labels, true_labels)
        if do_f1:
            f1 = f1_score(true_labels, pred_labels, average='binary',pos_label=1)
            pr_score = precision_score(true_labels, pred_labels, average="binary",pos_label=1)
            re_score = recall_score(true_labels, pred_labels, average="binary", pos_label=1)
        else:
            f1 = 0.0
            pr_score = 0.0
            re_score = 0.0
        
        return {"cls_acc": cls_score ,"cls_f1":f1, "cls_pr":pr_score, "cls_re":re_score}