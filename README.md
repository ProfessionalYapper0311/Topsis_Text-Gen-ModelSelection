# TOPSIS for Pre-Trained Text Generation Models

## Overview

This project applies the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method to select the best pre-trained model for text generation tasks. We evaluate models based on performance and efficiency metrics.

## Models Evaluated

* Llama-3-8B
* Mistral-7B
* Gemma-7B
* Falcon-7B
* GPT-2-1.5B

## Evaluation Criteria

| Criterion | Description | Impact |
| :--- | :--- | :--- |
| Parameter Size | Storage and memory size of the model in billions | Cost (-) |
| Context Window | Maximum token length for input prompts | Benefit (+) |
| MMLU Score | General knowledge benchmark accuracy (%) | Benefit (+) |
| HumanEval Score | Coding benchmark accuracy (%) | Benefit (+) |
| Inference Latency | Time taken to generate per token (ms) | Cost (-) |

## Results

The TOPSIS method calculated a score for each model. Higher scores indicate a better balance between high text generation capability and low resource usage.

| Model | Topsis Score | Rank |
| :--- | :--- | :--- |
| Llama-3-8B | 0.7828 | 1 |
| Gemma-7B | 0.5440 | 2 |
| Mistral-7B |0.5357 | 3 |
| Falcon-7B | 0.2888 | 4 |
| GPT-2-1.5B | 0.2439 | 5 |

*(Note: Llama-3-8B ranks high because it achieves significantly higher accuracy on complex benchmarks like MMLU and HumanEval without a massive increase in latency or parameter size.)*

## Visualization

<img width="813" height="510" alt="image" src="https://github.com/user-attachments/assets/3eb0640a-7334-46d7-b891-567e84c614d9" />


## How to Run

Install dependencies: 
```bash
pip install pandas numpy matplotlib
