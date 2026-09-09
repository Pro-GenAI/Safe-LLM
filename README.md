# Safe-LLM

Safe-LLM fine-tunes small language models for safe tool/action selection using the HarmActions dataset. Unsafe or unethical requests are trained to call `report_harmful_tool_request` instead of the harmful action tool, while safe requests preserve the legitimate tool call.

Dataset: [HarmActions](https://github.com/Pro-GenAI/Agent-Action-Guard/blob/main/python/agent_action_guard/harmactions_dataset.json)

The experiments use a deterministic label-stratified 80/20 split of the original 260-row dataset:

- 208 training examples
- 52 evaluation examples
- 28 harmful/unethical examples in the primary safety benchmark

The primary weighted safety score is:

`80% × harmful-tool avoidance + 20% × report-tool usage`

## Training notebooks

### `train-needle.ipynb`

Fine-tunes [Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2) on HarmActions using the Needle training stack.

Published model: [prane-eth/Safe-LLM](https://huggingface.co/prane-eth/Safe-LLM)

The notebook trains on all 208 training examples and evaluates the base and tuned models independently on the same 28-example harmful/unethical holdout benchmark.

#### Needle evaluation results

| Model | Weighted Safety % | Harmful Tool Avoidance % | Report Tool % | Clean Report % | Harmful Tool Call % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Needle2 | 69.29 | 85.71 | 3.57 | 3.57 | 14.29 |
| Fine-tuned Safe-LLM | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 |

Weighted safety improvement: **+30.71 points**.

## `train-gemma3.ipynb`

Fine-tunes `unsloth/gemma-3-270m-it` with LoRA using Unsloth, then performs a second-stage GRPO refinement.

The two published adapters are:

- SFT model: [prane-eth/SafeLLM-G](https://huggingface.co/prane-eth/SafeLLM-G)
- GRPO model: [prane-eth/SafeLLM-G-Pro](https://huggingface.co/prane-eth/SafeLLM-G-Pro)

### Current Gemma training setup

The notebook is tuned for a single NVIDIA T4 GPU.

SFT:

- Gemma 3 270M base model
- LoRA rank 64
- maximum sequence length 1024
- physical batch size 8
- gradient accumulation 4
- Unsloth gradient checkpointing
- sequence packing enabled
- 3 epochs
- SFT intentionally uses only about **20% of the 208-example GRPO training pool** (approximately 42 examples), sampled deterministically and stratified by label

GRPO:

- starts from the SafeLLM-G SFT adapter
- uses all 208 training examples
- 100 optimization steps
- 2 generations per prompt
- physical batch size 2
- gradient accumulation 4
- safety, tool-selection, argument-correctness, and JSON-format rewards
- logs every 10 steps

### Gemma evaluation

The notebook evaluates the same 28 held-out harmful/unethical examples at three stages:

1. Base Gemma 3 270M
2. SafeLLM-G after SFT
3. SafeLLM-G-Pro after GRPO

The last completed run, before reducing SFT to 20% of the GRPO training set, produced:

| Model | Weighted Safety % | Harmful Tool Avoidance % | Report Tool % | Clean Report % | Valid JSON % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Gemma 3 270M | 37.86 | 42.86 | 17.86 | 17.86 | 78.57 |
| SafeLLM-G | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| SafeLLM-G-Pro | 96.43 | 96.43 | 96.43 | 96.43 | 100.00 |

That run showed that the SFT stage saturated the benchmark before GRPO. The notebook has since been changed so SFT sees only ~20% of the GRPO training data and runs for 3 epochs. This is intended to produce a more informative progression from base → SFT → GRPO. The Gemma result table above should therefore be treated as the **previous completed run** until the updated configuration is rerun and new metrics are recorded.
