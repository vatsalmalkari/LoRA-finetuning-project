# LoRA Fine-Tuning Project

Test whether LoRA and QLoRA can adapt a small instruction-tuned model to baseball questions on a free Colab T4 GPU. Measure accuracy, training time, memory, and inference speed. This extends the LoRA paper (Hu et al., 2021) by checking its claims at small scale and on a new domain.

Stats questions test a learned calculation, not baseball knowledge.

## Hypotheses

LoRA improves accuracy over the base model with a small fraction of trainable parameters.
Very low rank (1 to 8) is enough for this domain.
Adapting both q_proj and v_proj works better than adapting q_proj alone.
QLoRA reduces peak memory with a small accuracy cost.

## Model

Gemma 3 1B IT (google/gemma-3-1b-it)
Access requires accepting Google's Gemma terms on Hugging Face and setting a read token.

## Dataset

Training: 300 examples
Validation: 30 examples, used only to tune the learning rate. Never used for reported scores.
Test: 100 examples, held out before any training begins
Question types: multiple choice (rules and history), short factual answers (normalized exact match), and computed stats (answers calculated in code)
Format: chat-format JSON with user and assistant messages
Check for overlap between train and test before training.

## Methods

Primary experiments:
Base model
LoRA, rank 8, targets q_proj and v_proj
QLoRA, rank 8, targets q_proj and v_proj, 4-bit NF4

Module names follow Hugging Face conventions: q_proj, k_proj, v_proj, and o_proj correspond to Wq, Wk, Wv, and Wo in  paper.

## Settings

Learning rate: 2e-4, cosine schedule, 3% warmup
LoRA alpha: 16
LoRA dropout: 0.05
Epochs: 3
Effective batch size: 8
Max sequence length: 512
Optimizer: AdamW
Precision: fp16 on the T4, with fp32 fallback if losses become unstable
Seeds: 1, 100, 1000

## Metrics

Accuracy by question type and overall, as mean and standard deviation across seeds
Trainable parameters and percentage of total
Adapter file size
Training time and tokens per second
Peak GPU memory
Time to generate 64 tokens at batch size 1, averaged over 20 prompts

## Procedure

Run baselines and save outputs.
Run a pipeline check: train on 20 examples for one epoch.
Run LoRA and QLoRA with few seeds.
Run the rank and module comparisons if time allows.
Compile results and write the report.

## Example

Setup: Colab, T4 GPU, install libraries, log in to Hugging Face.

Data: Build training examples from Lahman stats with code, plus hand-written rules questions. Freeze 100 test items before training. Check for overlap.

Baseline: Run Gemma 3 1B on the test set, zero-shot and few shot. Save scores.

Pipeline check: Train on 20 examples for one epoch. Confirm the loss drops and scoring runs.

Train: LoRA (rank 8, q_proj and v_proj), then QLoRA (same, 4-bit). Use 300 examples, 3 epochs, learning rate 2e-4. Log time, memory, and accuracy.

Seeds: Repeat with few more seeds.

Compare: Report mean and standard deviation. Check if LoRA beats the baseline by more than the spread.

Write up: Describe data, methods, results, and the limit that stats questions test calculation, not baseball knowledge.