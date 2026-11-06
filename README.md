# LoRA-finetuning-project

Test whether LoRA and QLoRA can adapt a small instruction-tuned model to baseball questions on a free Colab T4 GPU. Measure accuracy, training time, memory, and inference speed. This extends the LoRA paper by checking its claims at small scale and on a new domain.

Hypotheses

LoRA improves accuracy over the base model with a tiny fraction of trainable parameters.
Very low rank (1 to 8) is enough for this domain.
Adapting both Wq and Wv works better than adapting Wq alone.
QLoRA reduces peak memory with a small accuracy cost, and the saving is larger on the 4B model.

Models

Gemma 3 1B IT as the primary model.
Gemma 3 4B IT for the QLoRA memory test.
Access requires accepting Google’s Gemma terms on Hugging Face and setting a read token.

Dataset

Baseball questions in three types: multiple choice on rules and history, short factual answers with normalized exact match, and computed stats questions with answers calculated in code.
About 300 training examples, 100 held-out test examples, and optionally 30 validation examples for tuning the learning rate.
Record the source and license of every item. Do not copy copyrighted rulebook text.
Store data in chat-format JSONL. Check for overlap between train and test before training.

Methods

Base model, zero-shot.
Base model, few-shot with 3 examples.
LoRA with rank 8 on Wq and Wv.
QLoRA with rank 8 on Wq and Wv, using 4-bit NF4.
Rank sweep at ranks 1, 2, 4, 16, and 64.
Ablation comparing Wq only with all four attention matrices.
DoRA at rank 8 as an extension, if supported.
QLoRA on Gemma 3 4B.
Full fine-tuning is cited from the paper, not run.

Settings

Learning rate 2e-4 with cosine schedule and 3% warmup.
LoRA alpha 16, dropout 0.05, 3 epochs, effective batch size 8, max length 512, AdamW.
Run 3 seeds per configuration: 42, 1337, 2024.
Use fp16 on the T4, with an fp32 fallback if losses become unstable.

Metrics

Accuracy by question type and overall, reported as mean and standard deviation across seeds.
Trainable parameters and percentage of total.
Adapter file size.
Training time and tokens per second.
Peak GPU memory.
Time to generate 64 tokens at batch size 1, averaged over 20 prompts.

Procedure

Set up the environment and confirm the GPU works.
Run baselines and save outputs.
Run a small pipeline check: train on 20 examples for one epoch.
Run LoRA and QLoRA with 3 seeds.
Run the rank sweep and ablation.
Run DoRA and the 4B QLoRA test if time allows.
Compile results and write the report.

Primary experiment
Model: Gemma 3 1B IT.
GPU: Free Colab T4.
Dataset: 500 training, 100 validation, 200 test.
Methods: zero-shot, 3-shot, LoRA, QLoRA.
LoRA rank: 8.
Targets: q_proj + v_proj.
Training: 3 epochs, effective batch 8, LR 2e-4.
Seeds: 42, 1337, 2024.
Metrics: accuracy, parameter count, training time, memory, inference latency.

Research experiments
Rank sweep: 1, 2, 4, 8, 16, 64.
Target module ablation: q, v, q+v, q+k+v+o.
Gemma 3 4B QLoRA memory and performance comparison.

Should answer:
Does LoRA improve baseball task performance over zero-shot and few-shot prompting?
Does rank 1–8 achieve comparable results to higher ranks?
Does targeting multiple attention projections improve adaptation?
Does QLoRA preserve accuracy while reducing memory?
How does model size affect the effectiveness and memory cost of adaptation?
Does fine-tuning improve generalization, or primarily memorization and output formatting?
How much does inference latency change with adapters?