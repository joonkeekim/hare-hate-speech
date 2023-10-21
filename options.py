from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments, GenerationConfig

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    tasks : Optional[str] = field(
        default="hate", metadata={"help": "Choose task within hate(HateXplain), implicit(Implicit Hate)"},
    )

    wandb_name : Optional[str] = field(
        default="hare", metadata={"help": "name of the wandb run"},
    )
    wandb_entity : Optional[str] = field(
        default="hare", metadata={"help": "group name of the wandb run"},
    )
    wandb_group : Optional[str] = field(
        default="hare", metadata={"help": "group name of the wandb run"},
    )
    reasoning:  bool = field(
        default=False, metadata={"help": "Whether to use rationale."}
    )
    cti:  bool = field(
        default=False, metadata={"help": "Whether to use rationale."}
    )
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "Test file to final evaluation."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                          "the model."})
    do_test: Optional[bool] = field(default=False, metadata={
                                    "help": "If set, evaluates the test performance."})
    compute_time: Optional[bool] = field(
        default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(
        default=False, metadata={"help": "if set, measures the memory"})
    eval_all_at_last: bool = field(
        default=False,
        metadata={
            "help": "evaluate all checkpoints on all tasks at the last"
        },)
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate to calculate generative metrics."}
    )
    generation_config = GenerationConfig(
        max_new_tokens=512, do_sample=True, top_k=20,
    )