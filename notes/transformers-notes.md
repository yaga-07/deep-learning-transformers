# deep-learning-transformers
Learning Notes, Code and files

Let’s break down what a **Transformer layer** is and how it works at an atomic level in a simple, clear way.

---

### **1. What is a Transformer Layer Made Of?**

A **Transformer layer** has two main parts:

#### **A. Multi-Head Self-Attention**

* It helps the model **focus on different words (or tokens)** in the input when processing each word.
* It computes relationships **between every pair of words**, so the model understands context.

#### **B. Feed-Forward Neural Network (FFN)**

* After attention, the model processes each word independently using a small MLP (2 linear layers with a non-linearity like ReLU or GELU).

And in between those, we have:

* **Add & LayerNorm**: Helps stabilize and normalize outputs.
* **Residual Connections**: Helps gradients flow (like skip connections).

---

### **2. Atomic-Level View: Step-by-Step**

Let’s say the input is a sentence:

> “The cat sat on the mat”

This is what happens inside a single Transformer layer:

#### **Step 1: Input Embedding**

Each word becomes a vector (from a word embedding or token embedding).

```python
Input: ["The", "cat", "sat", "on", "the", "mat"]
Embeddings: [x1, x2, x3, x4, x5, x6]  # Each xi is a vector
```

#### **Step 2: Self-Attention (Core Idea)**

For **each word**, we ask:

> “How much should I pay attention to all the other words?”

This is done by computing:

* **Query (Q)**, **Key (K)**, and **Value (V)** vectors for each word:

  ```python
  Q = x @ Wq
  K = x @ Wk
  V = x @ Wv
  ```
* Compute attention scores:

  ```python
  score = Q @ K.T / sqrt(d)
  weights = softmax(score)
  output = weights @ V
  ```

Each word now gets a **contextualized vector** — it now **knows about the other words**.

#### **Step 3: Multi-Head Attention**

* This self-attention is done **multiple times in parallel** (say, 8 heads).
* Each head focuses on different kinds of relationships (like subject-object, or tense).

#### **Step 4: Add & Layer Norm**

* Add input to attention output (residual)
* Normalize to prevent exploding/vanishing gradients.

#### **Step 5: Feed-Forward Network**

* For each word independently:

  ```python
  out = Linear1(x)
  out = ReLU(out)
  out = Linear2(out)
  ```

#### **Step 6: Add & Layer Norm Again**

* Another residual connection and normalization.

---

### **3. Putting It Together**

A **Transformer Layer** processes inputs like this:

```
Input embeddings
   │
[Multi-Head Attention]
   │
[Add & Norm]
   │
[Feed-Forward (MLP)]
   │
[Add & Norm]
   ↓
Output embeddings (context-aware)
```

Each layer **enriches the representation**, so higher layers understand more abstract meanings.

---

### **4. Simple Example**

Let’s say you're processing the sentence:

> "The cat sat on the mat."

You want the model to understand that “cat” is the one who “sat”.

Self-attention will help the word “sat” focus on “cat”, even though there are other words in between. The attention weight between “sat” and “cat” will be **high**, so "sat" gets information about "cat".

This way, the model captures **who did what to whom** — essential for understanding language.

---
---

### **1. Stacking Transformer Layers: The Big Picture**

A **Transformer model** is basically **a stack of identical Transformer layers** — say **6 layers** for the original Transformer, **12 for BERT-base**, and even **96+ for GPT-4**.

Each layer refines the representations it receives from the previous layer.

#### **High-Level Structure**

```
Input tokens
   ↓
Token Embeddings + Positional Encoding
   ↓
┌──────────────────────────────┐
│   Transformer Layer 1        │
└──────────────────────────────┘
        ↓
┌──────────────────────────────┐
│   Transformer Layer 2        │
└──────────────────────────────┘
        ↓
        .
        .
        ↓
┌──────────────────────────────┐
│   Transformer Layer N        │
└──────────────────────────────┘
        ↓
Final output (for classification, generation, etc.)
```

---

### **2. What Happens as We Stack?**

Let’s say our input sentence is again:

> “The cat sat on the mat.”

Each word is embedded into a vector, then passed through the stack:

* **Layer 1**: Each token learns immediate context (e.g., “cat” attends to “the” and “sat”)
* **Layer 2**: Now “cat” understands it is part of a subject doing an action.
* **Layer 3**: “Sat” starts understanding deeper grammatical structure.
* ...
* **Layer N**: High-level meaning is captured — like sentence intent, entities, relationships.

So, **deeper layers build more abstract understanding**.

---

### **3. Example: Stacking in BERT**

BERT-base has:

* **12 Transformer layers**
* **Each layer has:**

  * 12 attention heads
  * A feed-forward network of size 3072
* Final output: a context-rich embedding for each token

These outputs can be used for:

* **Classification**: Take `[CLS]` token output
* **QA**: Predict start and end positions
* **NER**: Label each token output

---

### **4. Visualization (Text Form)**

Here’s a text-based sketch of a stack:

```
Token Embeddings + Positional Encoding
        ↓
  ┌────────────────────────┐
  │  Transformer Layer 1   │
  └────────────────────────┘
        ↓
  ┌────────────────────────┐
  │  Transformer Layer 2   │
  └────────────────────────┘
        ↓
  ┌────────────────────────┐
  │  Transformer Layer 3   │
  └────────────────────────┘
        ↓
        ...
        ↓
  ┌────────────────────────┐
  │  Transformer Layer N   │
  └────────────────────────┘
        ↓
Contextual embeddings for all tokens
```

---

### **5. Why Stack?**

Stacking helps the model to:

* **Capture long-range dependencies** (deep layers = broad context)
* **Learn abstract patterns** (grammar, semantics, logic)
* **Encode hierarchical structure of language** (phrase → clause → sentence meaning)

---
---

### **How Transformers Handle Arbitrary-Length Input**

The key components that **enable Transformers to handle variable-length sequences** are:

---

### **1. Input as a Sequence of Tokens (Not Fixed Size Vectors)**

Unlike CNNs or some older models that need fixed-size inputs, Transformers process **sequences** of **token embeddings**:

* You can feed in a sequence of **any length** (within hardware or model limits), like:

  ```
  ["The", "cat", "sat"] → 3 tokens
  ["The", "cat", "sat", "on", "the", "mat"] → 6 tokens
  ```

There’s **no hardcoded input size** — the input is treated as a list of vectors.

---

### **2. Self-Attention is Size-Agnostic**

Self-attention operates on **all tokens at once**, by computing:

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv

Attention = softmax(Q @ K.T / sqrt(d)) @ V
```

* `X` is the sequence of input vectors.
* Whether you have 3, 30, or 300 tokens, it just makes a larger attention matrix.
* There’s **no need to pad or truncate**, unless the model or training setup requires fixed batch sizes.

So self-attention naturally scales to variable lengths — it just computes relationships between however many tokens there are.

---

### **3. Positional Encoding Adds Sequence Information**

Since attention is **position-agnostic** (it doesn’t know the order of tokens), Transformers add **positional encodings** to input embeddings:

```python
Final Input = TokenEmbedding + PositionalEncoding
```

* These encodings inject **order awareness** into the model.
* They’re generated in a way that works for any length (usually via sine/cosine functions or learned embeddings).

---

### **4. Model Is Fully Parallel and Permutation-Friendly**

Because of how self-attention works:

* The model doesn’t need to unroll or scan the sequence one step at a time (like RNNs).
* It computes everything in parallel, regardless of the sequence length.

---

### **Real Limitations?**

While the design supports arbitrary length **in theory**, in **practice**:

* You’re limited by **memory** (since self-attention scales as O(n²) in sequence length).
* Most models set a **maximum length** (e.g., 512 for BERT, 2048 or 8192 for GPT).

But **nothing in the architecture itself fixes the length** — it’s flexible by design.

---

### **TL;DR: Why Transformers Support Arbitrary Input Lengths**

| Feature                  | Role in Arbitrary Length    |
| ------------------------ | --------------------------- |
| Token sequence input     | No fixed size required      |
| Self-attention mechanism | Operates on any length      |
| Positional encoding      | Preserves order flexibly    |
| No recurrence or padding | Fully parallel and flexible |

---
Let’s now **build the full Transformer encoder block** step by step, with:

---

## **1. Tensor Shape Flow in Transformer Encoder Block**

Let’s assume:

* `B` = batch size
* `n` = number of tokens
* `d_model` = model dimension (e.g., 512)
* `h` = number of heads (e.g., 8)
* `dk = dv = d_model / h`

### **Full Encoder Block:**

```
Input: X  [B, n, d_model]

Step 1: Multi-Head Self Attention
  ┌────────────┐
  │ Linear W_q │ → [B, n, h, dk]
  │ Linear W_k │ → [B, n, h, dk]
  │ Linear W_v │ → [B, n, h, dv]
  └────┬───────┘
       │
  Scaled Dot-Product Attention per head
       ↓
  Concatenate heads → [B, n, h * dv] = [B, n, d_model]
  Final Linear Projection → [B, n, d_model]

Step 2: Add & Norm
  LayerNorm(X + AttentionOut)

Step 3: Feed-Forward Network (FFN)
  ┌─────────────────────────────┐
  │ Linear(d_model → d_ff)      │
  │ ReLU (or GELU)              │
  │ Linear(d_ff → d_model)      │ → [B, n, d_model]
  └─────────────────────────────┘

Step 4: Add & Norm
  LayerNorm(X + FFNOut)
```

---

## **2. PyTorch Code: Full Transformer Encoder Block (Custom)**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, n, _ = x.shape
        # Project
        Q = self.W_q(x).view(B, n, self.num_heads, self.dk).transpose(1, 2)  # [B, h, n, dk]
        K = self.W_k(x).view(B, n, self.num_heads, self.dk).transpose(1, 2)
        V = self.W_v(x).view(B, n, self.num_heads, self.dk).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / self.dk ** 0.5  # [B, h, n, n]
        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ V  # [B, h, n, dk]

        # Concatenate heads
        context = context.transpose(1, 2).reshape(B, n, self.d_model)  # [B, n, d_model]
        return self.W_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # or GELU
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ff(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Multi-Head Attention + Residual + Norm
        attn_out = self.self_attn(x)
        x = self.norm1(x + attn_out)

        # Feedforward + Residual + Norm
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x
```

---

## **3. Example Usage**

```python
B = 2         # batch size
n = 5         # number of tokens
d_model = 64
num_heads = 8
d_ff = 256

x = torch.randn(B, n, d_model)
encoder = TransformerEncoderBlock(d_model, num_heads, d_ff)

out = encoder(x)
print("Output shape:", out.shape)  # [B, n, d_model]
```

---
Let's now build the **Transformer Decoder Block** and then combine it with the encoder to complete the **full Transformer architecture**. 

---

## **1. Transformer Decoder Block (with shapes)**

A Transformer decoder block is slightly more complex than the encoder block due to **two attention layers**:

```
Input: Target sequence embeddings → [B, t, d_model]
Encoder output (context)         → [B, s, d_model]

Step 1: Masked Multi-Head Self Attention (decoder attends to previous tokens only)
        Input: [B, t, d_model]
        Output: [B, t, d_model]

Step 2: Add & Norm

Step 3: Encoder-Decoder Cross Attention
        Q = decoder input → [B, t, d_model]
        K, V = encoder output → [B, s, d_model]
        Output: [B, t, d_model]

Step 4: Add & Norm

Step 5: Feed-Forward Network
        → [B, t, d_model]

Step 6: Add & Norm
```

* `B`: Batch size
* `t`: Target sequence length
* `s`: Source sequence length (input)
* `d_model`: Embedding/model dimension

---

## **2. Code: Transformer Decoder Block (Custom PyTorch)**

### **Multi-Head Attention with Optional Cross Attention and Masking**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, nq, _ = q.shape
        nk = k.size(1)

        Q = self.W_q(q).view(B, nq, self.num_heads, self.dk).transpose(1, 2)  # [B, h, nq, dk]
        K = self.W_k(k).view(B, nk, self.num_heads, self.dk).transpose(1, 2)  # [B, h, nk, dk]
        V = self.W_v(v).view(B, nk, self.num_heads, self.dk).transpose(1, 2)  # [B, h, nk, dk]

        scores = Q @ K.transpose(-2, -1) / self.dk ** 0.5  # [B, h, nq, nk]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ V  # [B, h, nq, dk]
        context = context.transpose(1, 2).reshape(B, nq, self.d_model)  # [B, nq, d_model]

        return self.W_o(context)
```

---

### **Decoder Block**

```python
class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + x2)

        # Cross Attention (Decoder attends to encoder outputs)
        x2 = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + x2)

        # Feedforward
        x2 = self.ffn(x)
        x = self.norm3(x + x2)

        return x
```

---

## **3. Full Transformer (Encoder + Decoder)**

```python
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size_src, vocab_size_tgt, max_len=512):
        super().__init__()
        self.embedding_src = nn.Embedding(vocab_size_src, d_model)
        self.embedding_tgt = nn.Embedding(vocab_size_tgt, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(d_model, vocab_size_tgt)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        B, s_len = src.shape
        B, t_len = tgt.shape

        src_embed = self.embedding_src(src) + self.pos_encoding[:, :s_len, :]
        tgt_embed = self.embedding_tgt(tgt) + self.pos_encoding[:, :t_len, :]

        enc = src_embed
        for layer in self.encoder_layers:
            enc = layer(enc)

        dec = tgt_embed
        for layer in self.decoder_layers:
            dec = layer(dec, enc, tgt_mask)

        return self.output_linear(dec)
```

---

### **Optional Mask Generator for Causal Attention**

```python
def generate_subsequent_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)  # [1, 1, t, t]
```

---

## **4. Final Flow Summary (Encoder + Decoder)**

```
src → embedding + pos → encoder blocks → enc_out
tgt → embedding + pos → decoder blocks (with enc_out + masks) → logits
```

---

Let’s dive into how the Transformer architecture can be extended to handle various modalities and explore key variants developed for different applications. 

---

# 🔄 Extending Transformers to Multiple Modalities

Transformers were originally designed for text, but their flexible architecture makes them applicable across different data types:

| Modality      | Example Applications                   | Transformer Variant                        |
| ------------- | -------------------------------------- | ------------------------------------------ |
| 📝 Text       | Language modeling, translation         | Standard Transformer, BERT, GPT            |
| 🖼️ Vision    | Image classification, object detection | Vision Transformer (ViT), Swin Transformer |
| 🎧 Audio      | Speech recognition, music generation   | Audio Spectrogram Transformer              |
| 🎥 Video      | Action recognition, video captioning   | TimeSformer, Video Swin                    |
| 🎮 RL/Game    | Decision making, game agents           | Decision Transformer                       |
| 🧠 Multimodal | VQA, Image Captioning, RAG, VLMs       | CLIP, Flamingo, Perceiver, PaLM-E          |

---

## 1. 📷 Vision Transformer (ViT)

### Idea:

Treat image patches like word tokens.

### Diagram:

```markdown
Image (H x W x 3)
↓
Split into patches → (N, P, P, 3)
↓
Flatten patches → (N, P²·3)
↓
Linear projection → (N, D)
↓
+ Positional Encoding
↓
Standard Transformer Encoder
↓
[CLS] token output → classification
```

### Shapes:

* Input Image: `3 x 224 x 224`
* Patches: `16 x 16 → 196 patches`
* Flattened: `196 x (16*16*3) = 196 x 768`
* After projection: `196 x D` (e.g. `768`)

### Code (Patch Embed + Transformer Encoder):

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

# Sample usage
img = torch.randn(2, 3, 224, 224)
patch_embed = PatchEmbed()
tokens = patch_embed(img)  # [2, 196, 768]
```

---

## 2. 🎧 Audio Transformer

### Idea:

Convert waveform → spectrogram → image-like patches → transformer.

### Spectrogram:

Transform audio signal into a 2D frequency-time representation.

```markdown
Audio (waveform)
↓
Spectrogram (T x F)
↓
Treat like image patches
↓
Transformer encoder
```

### Code (Waveform to Spec + Patch Embed):

```python
import torchaudio

waveform, sr = torchaudio.load("audio.wav")
spec_transform = torchaudio.transforms.MelSpectrogram()
spec = spec_transform(waveform)  # [1, Mel, Time]

# Treat as image for PatchEmbed
spec = spec.unsqueeze(0)  # [B=1, 1, Mel, Time]
```

Then pass through the same `PatchEmbed` as in ViT.

---

## 3. 🧠 Multimodal Transformers (e.g. CLIP, Flamingo)

### CLIP:

Align image and text representations.

```markdown
Image → ViT Encoder → z_img
Text → Text Transformer → z_text
↓
Cosine similarity(z_img, z_text)
↓
Contrastive loss (bring matching closer, others apart)
```

### Code (Simplified Encoder):

```python
class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size, dim, depth):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8),
            num_layers=depth)
        self.pool = lambda x: x.mean(dim=1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.pool(x)

# Similar encoder for ViT can be reused
```

---

## 4. 🕹️ Decision Transformer

### Idea:

Transformers for Reinforcement Learning (RL).

```markdown
[state_1, action_1, return_1,
 state_2, action_2, return_2,
 ..., state_t]
↓
Autoregressive transformer
↓
Predict next action
```

### Input Format:

```python
# Embed state, action, return separately and concatenate
x = [embed_r1, embed_s1, embed_a1, embed_r2, embed_s2, ...]
```

This lets the transformer learn behavior from returns instead of Q-learning.

---

## 5. 🧠 Perceiver / Perceiver IO (Universal Modality Handler)

### Idea:

Use a latent bottleneck that attends to inputs.

```markdown
Input: very large (e.g., image, video, point cloud, audio)
↓
Cross-attention → latent array (small)
↓
Repeated self-attention inside latent array
↓
Cross-attend to output queries (optional)
↓
Output
```

### Benefits:

* Handles very long or large inputs (efficient!)
* Modality-agnostic

---

## 🔚 Summary Table

| Model                | Modality      | Key Mechanism                    |
| -------------------- | ------------- | -------------------------------- |
| ViT                  | Vision        | Patch embedding, encoder only    |
| Audio Transformer    | Audio         | Spectrogram + patch embedding    |
| CLIP                 | Vision + Text | Dual encoders + contrastive loss |
| Decision Transformer | RL            | Return-conditioned generation    |
| Perceiver            | All           | Latent bottleneck, scalable      |

---

# 🧠 Understanding What to Use as the Query in Cross-Modal Attention

---

## 🌐 What is Cross-Modal Attention?

Cross-modal attention allows **one modality to attend to another**. It’s often used to:

* Fuse information from different sources (e.g., text and image)
* Answer questions based on visual inputs (VQA)
* Generate captions based on images
* Align modalities in contrastive learning (like CLIP)

---

## 🔍 What is Query, Key, and Value in Attention?

In the attention mechanism:

```
Attention(Q, K, V) = softmax(QKᵀ / √d) × V
```

* **Query (Q)**: What we're trying to *understand* or *focus from*
* **Key (K)**: What we're *looking at* to decide what’s relevant
* **Value (V)**: What we’ll *retrieve* based on the relevance scores

---

## 🎯 So… What to Use as Query?

It depends on **which modality you want to enrich or inform**.

---

### 📝 Example 1: Text Attending to Image (Text Queries Image)

**Use Case**: Text wants to extract or ground meaning from an image (e.g., Visual Question Answering)

| Role | Tensor          | Meaning                              |
| ---- | --------------- | ------------------------------------ |
| Q    | Text embeddings | Text is trying to understand image   |
| K    | Image patches   | Used to compute relevance            |
| V    | Image patches   | Provide actual content to be fetched |

```python
# Q: Text [B, T, D]
# K, V: Image patches [B, N, D]
attn_weights = softmax(Q @ K.T / sqrt(D))  # [B, T, N]
contextual_text = attn_weights @ V         # [B, T, D]
```

---

### 🖼️ Example 2: Image Attending to Text (Image Queries Text)

**Use Case**: Image wants context from text (e.g., generating captions or matching image with description)

| Role | Tensor          | Meaning                            |
| ---- | --------------- | ---------------------------------- |
| Q    | Image patches   | Image is trying to understand text |
| K    | Text embeddings | Used to compute relevance          |
| V    | Text embeddings | Provide the retrieved context      |

```python
# Q: Image [B, N, D]
# K, V: Text [B, T, D]
attn_weights = softmax(Q @ K.T / sqrt(D))  # [B, N, T]
contextual_image = attn_weights @ V        # [B, N, D]
```

---

### 🤝 Example 3: Bidirectional (Both Modalities Attend to Each Other)

Used in models like **LXMERT**, **UNITER**, or **ViLBERT**:

* One layer where text queries image
* Another layer where image queries text
* Results are fused together or used downstream

---

## 🔄 When to Use Which?

| Scenario                              | Query          | Why?                                                   |
| ------------------------------------- | -------------- | ------------------------------------------------------ |
| Visual Question Answering (VQA)       | Text           | Text needs to find relevant visual context             |
| Image Captioning                      | Image          | Image needs language cues to generate words            |
| CLIP-style Embedding                  | Both separate  | No attention across modalities (uses contrastive loss) |
| Visual Grounding (e.g., pointing)     | Text           | Text wants to highlight a region in the image          |
| Cross-modal Fusion for Classification | Either or Both | Fuse features from both for downstream tasks           |

---

## 📊 Visual Representation

```plaintext
           [Text Embeddings]  --> Q
                  |
                  V
      Attention(Q=text, K=image, V=image)
                  |
                  V
       Output: text enriched with visual info
```

or

```plaintext
           [Image Patches] --> Q
                  |
                  V
      Attention(Q=image, K=text, V=text)
                  |
                  V
       Output: image enriched with text info
```

## ✅ Summary

| Modality that needs context | Use it as **Query** |
| --------------------------- | ------------------- |
| Text needs visual info      | Text = Q            |
| Image needs text grounding  | Image = Q           |
| Mutual fusion               | Alternate Q roles   |

---