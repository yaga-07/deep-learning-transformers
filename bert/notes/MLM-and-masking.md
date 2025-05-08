# 📚 Masked Language Modeling (MLM) – Internal Data Flow & Attention Mask Role

---

## 🧠 What is MLM?

Masked Language Modeling trains models (e.g., BERT) to predict randomly masked tokens in an input sequence.

**Example:**

```text
Original:   I love deep learning.
Masked:     I love [MASK] learning.
Target:     I love deep learning.
```

---

## 🧩 Key Components in MLM Training

| Element          | Purpose                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| `input_ids`      | Tokenized input sequence with `[MASK]` tokens replacing some originals |
| `labels`         | Original token IDs; `-100` for positions we don't compute loss on      |
| `attention_mask` | 1s for actual tokens, 0s for padding — controls attention computation  |

---

## 🧪 Example: Batch with Padding

```python
# Two input sentences
tokens_1 = ["[CLS]", "I", "love", "pizza", ".", "[SEP]"]
tokens_2 = ["[CLS]", "NLP", "is", "fun", "[SEP]"]

# After tokenization and masking
input_ids = [
  [101, 1045, 2293, 103,   1012, 102,   0],     # [MASK] replaces "pizza"
  [101, 17953, 2003, 103,  102,   0,     0]     # [MASK] replaces "fun"
]

attention_mask = [
  [1,   1,    1,    1,     1,    1,    0],
  [1,   1,    1,    1,     1,    0,    0]
]

labels = [
  [-100, -100, -100, 10733, -100, -100, -100],  # Predict "pizza"
  [-100, -100, -100, 4569,  -100, -100, -100]   # Predict "fun"
]
```

---

## 🔄 What Happens Internally?

1. **Embedding Lookup**
   `input_ids → embeddings`

2. **Attention Score Computation**
   Self-attention scores are computed between all token positions.

3. **Attention Mask Applied**

   * `attention_mask == 1`: keep score
   * `attention_mask == 0`: mask out (set score to `-inf`)
   * Prevents padded tokens from affecting attention.

4. **Masked Token Prediction**

   * `[MASK]` token's representation is used to predict the original word.
   * Loss is only computed for positions where `labels != -100`.

---

## 🧠 Why Attention Mask Matters

| Purpose                                    | Impact                                                     |
| ------------------------------------------ | ---------------------------------------------------------- |
| Ignore padding in attention                | Prevents model from using `[PAD]` tokens as context        |
| Enables batching of variable-length inputs | Efficient and consistent training                          |
| Not tied to `[MASK]` token                 | `[MASK]` is part of the input, not excluded from attention |

---

## ✅ Summary

* `attention_mask`: Controls **attention scope** (not loss or masking).
* `labels`: Controls **loss calculation** (via `-100` mask).
* Padding is needed for batching; attention mask ensures it doesn't interfere.