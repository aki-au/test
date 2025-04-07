# MMLU Benchmark (Hugging Face Edition)

**Welcome!** This repo is a modified fork of the original [MMLU benchmark](https://github.com/hendrycks/test), updated to support open-source Hugging Face models like `google/gemma-3b-it` and `stabilityai/stable-code-3b`.

Whether you're testing your favorite LLM or writing a Medium post while Modern Family plays in the background and do some benchmarking, this repo makes it easier to evaluate how well these models perform on knowledge-intensive tasks.

---

## What's MMLU?

**MMLU (Massive Multitask Language Understanding)** is a benchmark designed to test language models across **57 subjects**

For a deep dive, check out my blog post:  
[Paper Breakdown #1 – MMLU: LLMs Have Exams Too!](https://medium.com/@alakarthika01/paper-breakdown-1-mmlu-llms-have-exams-too-a-post-on-benchmarking-a66630dfd2a6)

---

## What's in this repo?

- **Modified `evaluate.py`** to work with Hugging Face models  
- A random subset of **20 subjects** from MMLU (because Colab runtimes aren't infinite)  
- Scripts to run few-shot evaluation  


---

## [Quickstart is in the Colab notebook](https://colab.research.google.com/drive/1gHRLSgwstosutmww3onQlimSYpJlT8WA?usp=sharing)


   You can swap in any Hugging Face causal LM (`AutoModelForCausalLM` compatible).

---

## Models Used in My Tests

| Model                        | Description                            |
|-----------------------------|----------------------------------------|
| `google/gemma-3b-it`        | General-purpose instruction-tuned LLM  |
| `stabilityai/stable-code-3b` | Code-first model, tested just for fun  |

[Results](https://drive.google.com/drive/folders/19IihpcyHIioSxj_oA38nVrX4S-ubKWX1?usp=sharing)
---

## Subjects Tested (20)

`Abstract Algebra`, `Anatomy`, `College Biology`, `College Chemistry`, `College Mathematics`, `Global Facts`, `High School Biology`, `High School Computer Science`, `High School Government and Politics`, `High School World History`, `Human Sexuality`, `Management`, `Medical Genetics`, `Miscellaneous`, `Moral Disputes`, `Professional Accounting`, `Public Relations`, `Sociology`, `Virology`, `World Religions`

---

## Credits & References

- [Original MMLU Benchmark](https://github.com/hendrycks/test)  
- [LiuYiuWei LLM Evaluation](https://github.com/LiuYiuWei/LLM-Evaluation)  
- [My Medium Post on MMLU](https://medium.com/@alakarthika01/paper-breakdown-1-mmlu-llms-have-exams-too-a-post-on-benchmarking-a66630dfd2a6)

---

## Final Thoughts

LLMs are smart, but they're not magic. This repo exists to help you measure just how smart (or not) they really are.

Got suggestions? Found a bug? Want to run it on another model? Open an issue or shoot me a message. Let’s benchmark responsibly.
