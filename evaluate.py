import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from scipy.special import softmax

from crop import crop


choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = ["A", "B", "C", "D"]

    for i in range(test_df.shape[0]):
        # Build prompt
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # Crop prompt until it fits in the model's context window
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # Tokenize and get logits
        while True:
            try:
                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
                with torch.no_grad():
                    outputs = model(input_ids)
                logits = outputs.logits
                break
            except Exception as e:
                print("Error:", e)
                print("Retrying after pause...")
                time.sleep(1)
                continue

        # Get logits for next token
        next_token_logits = logits[0, -1, :]
        lprobs = []
        for ans in answers:
            ans_id = tokenizer(" " + ans, add_special_tokens=False).input_ids
            if len(ans_id) == 1:
                lprobs.append(next_token_logits[ans_id[0]].item())
            else:
                print(f"Warning: answer '{ans}' tokenized into multiple tokens. Using first token.")
                lprobs.append(next_token_logits[ans_id[0]].item() if ans_id else -100)

        pred_idx = int(np.argmax(lprobs))
        pred = answers[pred_idx]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return np.array(cors), acc, np.array(all_probs)

def main(args, model, tokenizer):
    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(args.data_dir, "test"))
        if "_test.csv" in f
    ])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    model_tag = args.model_name.replace("/", "_")  # For saving under safe folder name
    result_dir = os.path.join(args.save_dir, f"results_{model_tag}")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print(subjects)
    print(args)

    all_cors = []

    for subject in subjects:
        print(f"Running subject: {subject}")
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j, choice in enumerate(["A", "B", "C", "D"]):
            test_df[f"choice_{choice}_prob"] = probs[:, j]

        test_df.to_csv(os.path.join(result_dir, f"{subject}.csv"), index=False)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Weighted average accuracy across all subjects: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    main(args)

