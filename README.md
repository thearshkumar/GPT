# GPT Model Implementation

This repository contains a PyTorch implementation of a GPT (Generative Pre-trained Transformer) model, designed for efficient training and text generation. The implementation includes various modern techniques and optimizations for improved performance and scalability.

**Note**: This implementation is based on a tutorial by Andrej Karpathy. I acknowledge his valuable contribution to the field of AI and express my gratitude for making this knowledge accessible.

## Table of Contents
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Requirements](#requirements)

## Features

- Efficient GPT model implementation
- Distributed training support using PyTorch DistributedDataParallel (DDP)
- Mixed precision training with `torch.autocast`
- Gradient accumulation for large batch sizes
- Learning rate scheduling with warm-up and cosine decay
- Checkpoint saving and loading for resuming training
- Evaluation on validation set and Hellaswag dataset
- Text generation capabilities

## Model Architecture

The GPT model implementation includes the following key components:

1. **Transformer Blocks**: 
   - Self-attention mechanism (CausalSelfAttention)
   - Feed-forward neural network (MLP)
   - Layer normalization

2. **Embedding Layers**:
   - Token embedding (wte)
   - Positional embedding (wpe)

3. **Output Layer**:
   - Linear layer for token prediction (lm_head)

### Techniques Used

- **Self-Attention**: Implemented using `F.scaled_dot_product_attention` for efficient computation
- **Layer Normalization**: Applied before self-attention and feed-forward layers for stable training
- **Residual Connections**: Used in transformer blocks to facilitate gradient flow
- **Weight Tying**: The token embedding weights are shared with the output layer
- **Weight Initialization**: Custom initialization for improved training stability

## Training

The training process incorporates several advanced techniques:

1. **Distributed Training**: Supports multi-GPU training using PyTorch DistributedDataParallel (DDP)
2. **Mixed Precision**: Uses `torch.autocast` for faster training and reduced memory usage
3. **Gradient Accumulation**: Allows for effectively larger batch sizes
4. **Learning Rate Scheduling**: 
   - Warm-up phase
   - Cosine decay for the main training phase
5. **Gradient Clipping**: Prevents exploding gradients
6. **Checkpointing**: Saves model state and training progress for resuming training
7. **Evaluation**: 
   - Periodic evaluation on validation set
   - Hellaswag dataset accuracy measurement
8. **Custom Data Loading**: Efficient data loading using a custom `DataLoaderLite` class

## Usage

To use the trained model for text generation, follow these steps:

1. Ensure you have the required dependencies installed (see [Requirements](#requirements)).

2. Download the trained model checkpoint (e.g., `model_40000.pt`).

3. Run the `useModel.py` script:

   ```
   python useModel.py
   ```

4. The script will:
   - Load the model from the checkpoint
   - Prompt you to enter a text prompt
   - Generate multiple sequences based on your prompt using different random seeds

### Customization

You can modify the following parameters in the `useModel.py` script:

- `model_path`: Path to the model checkpoint
- `num_return_sequences`: Number of sequences to generate (default: 4)
- `max_length`: Maximum length of generated sequences (default: 100)
- `seeds`: List of random seeds for text generation (default: [42, 100, 200, 300])

### Example

```python
Enter your prompt: Once upon a time

Generating with seed=42
Sample 0: Once upon a time, there was a young girl named Sarah who lived in a small village in the mountains. She loved to explore the forest around her home, and one day she discovered a hidden cave. Inside the cave, she found a mysterious old book
Sample 1: Once upon a time, there was a little girl named Lily who lived in a small cottage in the woods. She loved to explore the forest and make friends with the animals. One day, while walking through the trees, she stumbled upon a magical clearing
Sample 2: Once upon a time, there was a young boy named Jack who lived in a small village at the edge of a vast forest. He was known for his curiosity and love of adventure. One day, while exploring the woods, he stumbled upon an old, overgrown path
Sample 3: Once upon a time, in a small village nestled in the mountains, there lived a young boy named Timothy. He was known for his kind heart and love of nature. One day, while walking through the forest, he stumbled upon a hidden glade filled with
--------------------------------------------------
```

## Requirements

- Python 3.7+
- PyTorch 1.13+
- tiktoken
- numpy

To install the required packages, run:

```
pip install torch tiktoken numpy
```
