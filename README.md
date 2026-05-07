# Prefix Tuning — PEFT on GPT-2

A from-scratch implementation of **Prefix Tuning**, a Parameter-Efficient Fine-Tuning (PEFT) technique, applied to GPT-2 for text generation.

---

## What is Prefix Tuning?

Instead of updating all model weights during fine-tuning, Prefix Tuning prepends a set of **trainable continuous vectors** (the prefix) to every layer's Key-Value attention. Only these prefix parameters are updated — the base model stays fully frozen.

```
Standard input:   [token1, token2, token3]
With prefix:      [P1, P2, P3, token1, token2, token3]  ← only P's are trained
```

| Method | What's trained | Trainable params |
|---|---|---|
| Full Fine-tuning | All weights | 100% |
| **Prefix Tuning** | **Prefix vectors per layer** | **~2.5%** |
| LoRA | Low-rank weight matrices | ~1% |
| Prompt Tuning | Input-layer prefix only | ~0.1% |

---

## Implementation

**`PrefixEncoder`**
- Learns a compressed embedding for the prefix tokens
- Projects it through a small MLP into Key and Value vectors for every transformer layer
- Returns a `DynamicCache` compatible with `transformers >= 4.40`

**`PrefixTuningModel`**
- Wraps frozen GPT-2 with the PrefixEncoder
- Extends the attention mask to cover prefix positions during training

```
Trainable parameters :   3,149,312  (~2.5%)
Frozen parameters    : 121,439,488  (GPT-2)
Saved checkpoint     :   ~2 MB  (prefix only, vs ~500 MB full model)
```

---

## Usage

Open `prefix_tuning_colab.ipynb` in Google Colab and run all cells.

**Requirements**
```
torch
transformers >= 4.40
```

---

## References

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation — Li & Liang, 2021](https://arxiv.org/abs/2101.00190)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
