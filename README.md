# high-perf-chatbot-torchscript
High-performance conversational AI chatbot built with PyTorch, TorchScript, and Luong attention. Optimized for fast inference, scripted for deployment, and trained on movie dialogs with hyperparameter tuning and profiling.

# TorchScript-Optimized Conversational AI Chatbot

> A high-performance, GPU-accelerated conversational AI chatbot trained on the Cornell Movie Dialog Corpus using a Sequence-to-Sequence architecture with Luong attention. Optimized via Weights & Biases hyperparameter sweeps and exported with TorchScript for deployment in non-Python environments.

---

## Project Summary

This project implements and optimizes a conversational chatbot trained on movie dialogues using PyTorch. It uses a Seq2Seq architecture with GRU layers and Luong-style attention, supports real-time greedy decoding, and is exportable via TorchScript for deployment.

The chatbot was optimized using Weights & Biases (W&B) hyperparameter sweeps and benchmarked using PyTorch Profiler to improve memory and compute efficiency. TorchScript conversion enables portable inference outside Python (e.g., mobile or C++ environments).

---

## Features

- Sequence-to-Sequence GRU architecture with Luong attention
- Trained on the Cornell Movie Dialogs Corpus
- Hyperparameter sweeps via Weights & Biases (W&B)
- GPU training with PyTorch & profiling
- TorchScript conversion (traced + scripted) for deployment
- Performance profiling via `torch.profiler`
- Exportable to CPU-compatible `.pt` model for inference in C++ (tested)

---

## Technical Stack

- **Python 3.11**, **PyTorch**
- **TorchScript** for model export
- **Weights & Biases** for hyperparameter tuning
- **torch.profiler** for performance analysis
- Jupyter Notebook for experimentation
- Google Colab (GPU backend) for training

---

## Model Training and Tuning

- Model: Seq2Seq with 2-layer GRU (encoder & decoder)
- Attention: Luong ("dot") attention mechanism
- Dataset: Cornell Movie Dialogs
- Embedding size: 500  
- Training iterations: 4000  
- Batch size: 64  
- Loss achieved: **2.88**

### Hyperparameter Sweep (W&B)

Tested 50 combinations with:
- Learning Rate: [0.0001, 0.00025, 0.0005, 0.001]
- Gradient Clipping: [0, 25, 50, 100]
- Decoder LR ratio: [1, 3, 5, 10]
- Optimizer: Adam / SGD
- Teacher Forcing: [0, 0.5, 1.0]

**Best configuration (jumping-sweep-17)**:
- Loss: 2.88  
- Clip: 100  
- LR: 0.0005  
- Optimizer: Adam  
- Teacher Forcing: 1.0  
- Decoder LR Ratio: 3.0

---

## TorchScript Conversion

Converted models for non-Python environments:
- Traced Encoder → `traced_encoder.pt`
- Traced Decoder → `traced_decoder.pt`
- Scripted GreedySearchDecoder → `scripted_searcher.pt`

```python
torch.jit.save(scripted_searcher, "scripted_chatbot_cpu.pth")
```

✔️ Fully compatible with TorchScript static graph  
✔️ Dynamic control flow handled via `torch.jit.script()`  
✔️ Exported for CPU (map_location="cpu") to support C++ deployment

---

## Performance Profiling

Used `torch.profiler` and Chrome Trace Viewer (`chrome://tracing`) to analyze:
- CUDA time
- Memory usage
- Execution bottlenecks

### ⏱️ Latency Comparison

| Model Type       | Inference Time | Speedup |
|------------------|----------------|---------|
| Native PyTorch   | 0.0651 sec     | 1x      |
| TorchScript      | 0.0519 sec     | 🚀 **1.25x** |

---

## 💬 Sample Responses

| Input             | Response                    |
|------------------|-----------------------------|
| hello            | hello . ? ? ? ?             |
| what's up?       | i want to talk . . !        |
| who are you?     | i am your father . . !      |
| where are you from? | i am not home . .        |

Note: Some responses reflect dataset bias and should not be used in production without moderation.

---

## Run Instructions

### Training (Colab)

```bash
python chatbot_train.py
```

### 🔬 Evaluation

```bash
python evaluate.py
```

### TorchScript Export

```bash
python export_torchscript.py
```

### Inference (Scripted)

```bash
python chatbot_infer.py
```

---

## What I Learned

- End-to-end ML pipeline: preprocessing → training → tuning → deployment  
- TorchScript conversion for portability  
- GPU profiling using `torch.profiler`  
- W&B for effective hyperparameter optimization  
- Latency benchmarking & model efficiency tuning

---

## Directory Structure

```
📂 chatbot.ipynb               # Main training + inference notebook
📂 nonPython_chatbot.cpp       # C++ inference attempt (TorchScript)
📂 chatbot_model.pt            # PyTorch model checkpoint
📂 scripted_searcher.pt        # Final TorchScript model (for deployment)
📂 traced_encoder.pt
📂 traced_decoder.pt
📂 libtorch-v2.1.0.zip         # LibTorch for Apple Silicon
📂 README.md                   # You're here!
```

---

## References

- [Weights & Biases](https://wandb.ai/)
- [TorchScript Docs](https://pytorch.org/docs/stable/jit.html)
- [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [LibTorch for Apple Silicon](https://github.com/mlverse/libtorch-mac-m1)

---

## Author

**Gopala Krishna Abba**  
[LinkedIn](https://linkedin.com/) • [W&B Project](https://wandb.ai/ga2664-new-york-university/chatbot)
