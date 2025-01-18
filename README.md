# Self-Supervised vs Supervised Vision Models: A Comparative Analysis

This project compares the performance and semantic learning capabilities of I-JEPA (a self-supervised vision model) with Google's Vision Transformer on CIFAR-10 classification tasks.

## Project Overview

This study explores:
- Transfer learning capabilities of I-JEPA vs. fine-tuned supervised ViT
- Classification performance on CIFAR-10
- Qualitative analysis of semantic understanding through class visualizations

## Key Results

![Classification Accuracies](./accuracies.png)

- **Supervised ViT**: 98.28% validation accuracy
- **Adapted I-JEPA**: 86.04% validation accuracy
- While the supervised ViT outperformed in classification, qualitative analysis suggests I-JEPA may better capture semantic meaning in its representations

## Repository Structure

- `notebooks/`: Jupyter notebooks for model training and evaluation
- `src/`: Source code for model implementations and utilities
- `paper/`: Detailed technical report of findings
- `results/`: Visualizations and numerical results
- `requirements.txt`: Project dependencies

## Getting Started

1. **Environment Setup**
```bash
pip install -r requirements.txt
```

2. **Data Preparation**
- CIFAR-10 is automatically downloaded through PyTorch
- I-JEPA checkpoint used: `proj/ijepa/checkpoint/IN1K-vit.h.14-300e.pth.tar`

## Visualization Examples

[Will add class visualization examples here]

## Implementation Details

### I-JEPA Adaptation
- Used target encoder for feature extraction
- Added layer normalization and MLP for classification
- Single epoch training with careful learning rate tuning

### Vision Transformer
- Fine-tuned Google's ViT model
- Modified output layer for CIFAR-10 classes
- Single epoch training for fair comparison

## Paper

For a detailed analysis of the methodology and findings, see the paper in the `paper/` directory.

## Future Work

Potential areas for further investigation:
- Unfreezing layers of I-JEPA target encoder
- Testing on other downstream tasks (segmentation, detection)
- Exploring curriculum learning strategies

## Citations

If you use this code or find our work helpful, please cite:
[Coming soon]

## License

[MIT License](LICENSE)