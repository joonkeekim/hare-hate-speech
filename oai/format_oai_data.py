# making request
import argparse
import pandas as pd
import os
import json

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default = './data/sbic/',
        type=str,
        help="the path of data"
    )

    parser.add_argument(
        "--data",
        default = 'sbic',
        type=str,
	    choices=['implicithate', 'sbic', 'hatexplain'],
        help="the path of data"
    )

    parser.add_argument(
        "--data_split",
        default = 'test',
        type=str,
	    choices=['train', 'valid', 'test'],
        help="the split of data"
    )
    
    parser.add_argument(
        "--raw_output",
        default = './rationales/raw.jsonl',
        type=str,
        help="raw output path"
    )
    parser.add_argument(
        "--filtered_output",
        default = './rationales/raw.jsonl',
        type=str,
        help="filtered output"
    )

    parser.add_argument(
        "--model",
        default = 'gpt-3.5-turbo',
        choices=['text-davinci-003', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301'],
        type=str,
        help="the model used for generation"
    )

    parser.add_argument(
        "--max_tokens",
        default = 256,
        type=int,
        help="number of max tokens for generation"
    )

    parser.add_argument(
        "--temperature",
        default = .99,
        type=float,
        help="temperature to adjust diversity"
    )

    parser.add_argument(
        "--num_choices",
        default = 1,
        type=int,
        help="number of generations for each prompt"
    )

    parser.add_argument(
        "--task_type",
        default = "request",
        choices=['request', 'filter'],
        type=str,
        help="the purpose of the call"
    )

    parser.add_argument(
        "--merge",
        default = False,
        type=bool,
        help="whether to incorporate the previous rationales"
    )

    parser.add_argument(
        "--attribute",
        default="pred",
        type=str,
        help="the attribute that is extracted"
    )

    parser.add_argument(
        "--request_form",
        default = './rationales/request.jsonl',
        type=str,
        help="the path of request form"
    )

    return parser

def find_rationale_by_id(rationales, i):
    for rationale in rationales:
        if rationale['id'] == i:
            return rationale
    return None  # or some suitable default

def request_API(args):
    """
        request the API to generate rationales for the given data
    """
    data_path = os.path.join(args.data_path, f'{args.data}_{args.data_split}.jsonl')
    with open(data_path, 'r') as file, open(args.request_form, "w") as r_file:
        for line in file:
            info = json.loads(line)
            ################ TODO : MAKE YOUR QUERY ##############
            # stage 1 extract COT rationales 
            query = f'Determine whether the following post is offensive, and why. First give a brief step-by-step explanation and then make a decision.\nPost: {info["post"]}\nAnswer:\nLet\'s explain it step by step. 1.'
            # stage 2 predict the label
            query += f'{info["COT"][5]} Therefore, if you must choose one between (A) Offensive and (B) Not offensive, the answer is :'
            
            # set the api request
            job = {"model": args.model, "n": args.num_choices, "temperature": args.temperature, "max_tokens": args.max_tokens,"messages": [{"role": "user", "content": f'{query}'}]}

            r_file.write(json.dumps(job) + '\n')


def filter_data(args):
    # Open your jsonl file and create a new file for the cleaned data
    ids = []
    with open(args.raw_output, 'r') as infile, open(args.filtered_output, 'w') as outfile:
        # Read each line from the input file
        for line in infile:
            index, json_data = line.split(" ", 1)
            json_data = json.loads(json_data)
            message = json_data[0]
            response = json_data[1]

            id = int(index)

            if id in ids:
                continue
            else:
                ids.append(id)

            # print(response)

            # Construct the cleaned data from api results
            cleaned_data = {
                "id": id,
                "model": message["model"],
                args.attribute: [choice['message']['content'] for choice in response['choices']]
            }

            # Write the cleaned data to the output file in jsonl format
            outfile.write(f"{json.dumps(cleaned_data)}\n")
    # if the option is merge, then the output rationales are incorporated into your original data
    if args.merge:
        rationales = []
        with open(args.filtered_output, 'r') as file:
            for line in file:
                info = json.loads(line)
                rationales.append(info)

        rationales = sorted(rationales, key=lambda x: int(x['id']))


        data_path = os.path.join(args.data_path, f'{args.data}_{args.data_split}.jsonl')
        # First, read the JSONL file and load each line into a list
        data = []
        with open(data_path, 'r') as file:
            for i, line in enumerate(file):
                json_data = json.loads(line)
                rationale = rationales[i]
                if rationale is not None:
                    if args.attribute in json_data:
                        if isinstance(json_data[args.attribute], list):
                            json_data[args.attribute].extend(rationale[args.attribute])
                        else:
                            json_data[args.attribute] = [json_data[args.attribute]] + rationale[args.attribute]
                    else:
                        json_data[args.attribute] = rationale[args.attribute]
                else:
                    raise ValueError(f"Rationale for id {i} not found")

                data.append(json_data)

        # Now, write the modified data back to the JSONL file
        with open(data_path, 'w') as file:
            for json_data in data:
                file.write(json.dumps(json_data) + '\n')




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.task_type == 'request':
        request_API(args)
    elif args.task_type == 'filter':
        filter_data(args)
            