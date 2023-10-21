import wandb

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from eval import Metrics
from trainer import ToxicTrainer
from data.implicit_dataset import IH2HateDataset, ImplicitCollator
from data.sbic_dataset import SBICDataset, SBICReasoningDataset, SBICCollator, SBIC2HateDataset
from options import ModelArguments, DataTrainingArguments, TrainingArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    wandb.init(project=data_args.tasks,
               entity=data_args.entity,
               group=data_args.wandb_group,
               name=data_args.wandb_name)
    
    # set seed
    set_seed(training_args.seed)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    metrics = Metrics(tokenizer, training_args.output_dir, training_args.zero_shot_test)
        
    if data_args.tasks == "im2hate":
        compute_metrics = metrics.compute_implicit_metrics
        implicit_dataset = IH2HateDataset
        test_dataset = implicit_dataset(data_args.test_data_file)
        data_collator = ImplicitCollator(tokenizer=tokenizer)
        
    elif data_args.tasks == "sbic2hate":
        compute_metrics = metrics.compute_sbic_metrics
        test_dataset = SBIC2HateDataset(data_args.test_data_file)
        data_collator = SBICCollator(tokenizer=tokenizer)

    trainer = ToxicTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix = 'test')
    
if __name__ == "__main__":
    main()