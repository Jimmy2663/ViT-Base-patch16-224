# Vision Transformer (ViT) for Soybean Leaf Disease Classification

A  **Vision Transformer (ViT)** implementation for multi-class soybean leaf disease Classification on the **SoyMulticlass Dataset**.

##  Project Overview

This project implements a  Vision Transformer that achieves state-of-the-art performance on the SoyMulticlass dataset containing four disease categories: Aerial Blight (AB), Bacterial Pustule (BP), Yellow Mosaic Virus (YMV), and Healthy leaves. 

### Key Achievements

- **4-Class Classification**: Distinguishes between Aerial Blight, Bacterial Pustule, Yellow Mosaic Virus, and Healthy leaves
- **Comprehensive Evaluation**: Top-1/Top-3 accuracy, precision, recall, F1-score, AUC metrics, and confusion matrices
- **10,869 Total Images**: Large-scale real-world agricultural dataset with 7,000+ training images
- **GPU-Optimized**: Leverages NVIDIA A100 GPU with 40GB memory for efficient training

##  Dataset Information

### SoyMulticlass Dataset Overview

The project uses a comprehensive dataset of real-world soybean leaf images collected during the 2024-2025 growing seasons. This is a large-scale, multi-class plant disease detection dataset specifically designed for evaluating deep learning models on agricultural disease classification.

#### Dataset Composition

| Category of Diseases | Total Images |
|---|---|
| **Yellow Mosaic Virus (YMV)** | 2,973 |
| **Bacterial Pustule (BP)** | 2,776 |
| **Aerial Blight (AB)** | 2,018 |
| **Healthy (HL)** | 3,102 |
| **Total Dataset** | **10,869** |

#### Disease Classes and Characteristics

| S. No. | Disease Name | Symptoms | Cause |
|---|---|---|---|
| 1 | **Aerial Blight** | Initial symptoms include water-soaked lesions that turn greenish-brown to reddish-brown and later brown or black | Fungal Infection |
| 2 | **Bacterial Pustule** | Symptoms consist of small, pale green spots with raised centres on leaves in the mid-to-upper canopy | Bacterial Infection |
| 3 | **Yellow Mosaic Virus** | Irregular green and yellow patches in leaves | Viral Infection |
| 4 | **Healthy** | No disease symptoms present | No Infection |

#### Dataset Statistics

- **Total Images**: 10,869 real-world soybean leaf images
- **Training Images**: ~7,000 (approximately 64% of total)
- **Validation + Test Images**: ~3,869 (approximately 36% of total)
- **Image Format**: JPEG/JPG with RGB channels
- **Image Resolution**: Standardized to 224×224 pixels during preprocessing
- **Collection Period**: 2024-2025 growing seasons
- **Collection Method**: Real-life field photographs of soybean plants
- **Label Distribution**: Balanced across all four disease classes

#### Why This Dataset Matters

This dataset addresses a critical agricultural challenge: early detection of soybean diseases is essential for crop protection and yield optimization. The multi-class nature of the dataset makes it suitable for evaluating transformer-based architectures that can capture complex disease patterns and distinguish between similar-looking symptoms.

##  Hardware & Software Specifications

### Computing Infrastructure

The experiments were conducted on high-performance computing infrastructure:

#### Hardware Specifications
| Component | Specification |
|---|---|
| **CPU** | AMD EPYC 7742 64C @ 2.25GHz |
| **CPU Cores** | 128 cores |
| **L3 Cache** | 256 MB |
| **System Memory (RAM)** | 1 TB |
| **GPU** | NVIDIA A100-SXM4 |
| **GPU Memory** | 40 GB |
| **Total GPUs per Node** | 8 |
| **Storage** | 10.5 PiB PFS-based storage |
| **Networking** | Mellanox ConnectX-6 VPI (infiniband HDR) |

#### Software Stack
| Component | Specification |
|---|---|
| **Operating System** | Ubuntu 20.04.2 LTS (DGXOS 5.0.5) |
| **CUDA Version** | 10.1 |
| **NVIDIA Driver Version** | 450.142.00 |
| **NVIDIA NGC Support** | https://ngc.nvidia.com/signin |
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | PyTorch with torchvision |
| **Libraries** | OpenCV, NumPy, Keras, TensorFlow, etc. |

This high-performance setup ensures rapid model training and experimentation, enabling efficient iteration on architecture designs.

##  Project Structure

```
├── experiment.py                # Main entry point - orchestrates full training pipeline
├── engine.py                    # Training/evaluation loop with comprehensive metrics
├── model_builder.py            # Custom Vision Transformer architecture
├── data_setup.py               # Data loading and preprocessing
├── utils.py                    # Utility functions for metrics, plotting, and model management
├── prediction_for_one.py       # Single image inference script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

##  File Descriptions

### `experiment.py`  **Main File**
The primary entry point that orchestrates the complete training workflow:
- Initializes the custom Vision Transformer model with configurable hyperparameters
- Sets up data loaders for train/validation/test splits from the SoyMulticlass dataset
- Runs the training loop for 100 epochs
- Generates comprehensive metrics, plots, and confusion matrices
- Saves the trained model and all evaluation results

**Key Hyperparameters:**
```python
NUM_EPOCHS = 100                # Training epochs
BATCH_SIZE = 32                 # Batch size
LEARNING_RATE = 0.00001         # Adam optimizer learning rate
D_MODEL = 768                   # Model dimensionality
N_HEADS = 12                    # Number of attention heads
N_LAYERS = 12                   # Number of transformer layers (KEY: can be reduced for better results)
IMAGE_SIZE = (224, 224)         # Input image size
PATCH_SIZE = (16, 16)           # Patch size (224/16 = 14×14 = 196 patches)
```

### `model_builder.py`
Implements the custom Vision Transformer architecture with modular components:

**Components:**
- **PatchEmbedding**: Converts images into patch embeddings via convolution
  - Conv2d layer with kernel=patch_size, stride=patch_size
  - Converts (B, 3, 224, 224) → (B, 196, 768)
  
- **PositionalEncoding**: Adds positional information and classification token
  - Learnable CLS token prepended to each sequence
  - Sinusoidal positional encodings for patch positions
  - Converts (B, 196, 768) → (B, 197, 768)
  
- **AttentionHead**: Single attention head for multi-head attention
  - Query, Key, Value projections
  - Scaled dot-product attention with softmax
  
- **MultiHeadAttention**: Parallel attention heads (default: 12 heads)
  - Concatenates outputs from all heads
  - Linear projection to original dimension
  
- **TransformerEncoderLayer**: Single transformer block
  - Layer normalization + Multi-head attention + Residual connection
  - Layer normalization + Feed-forward MLP (4×d_model) + Residual connection
  - GELU activation with 0.1 dropout
  
- **VisionTransformer**: Complete model
  - Stacks multiple transformer encoder layers (configurable depth)
  - Classification head: Linear → GELU → Linear → Softmax for 4 classes
  - **Architecture Modification**: Uses ModuleList of TransformerEncoderLayers for flexible depth

**Why Architecture Changes Matter:**
The standard ViT-Base uses 12 transformer layers, which is designed for large-scale datasets like ImageNet (1.2M images). For the SoyMulticlass dataset (~7K training images), this leads to overfitting. By reducing `N_LAYERS`, the model learns more generalizable features, hence the accuracy improvement from 47% → 85%.

### `engine.py`
Training and evaluation engine with comprehensive metric tracking:

- **train_step()**: Single epoch training
  - Forward pass, loss computation, backward pass, parameter updates
  - Top-1 and Top-3 accuracy calculation
  - Comprehensive metrics: Precision, Recall, F1-score, AUC
  
- **test_step()**: Validation/test evaluation
  - Inference mode (no gradient computation)
  - Same metrics as training for consistency
  
- **train_and_test()**: Complete training pipeline
  - Runs multiple epochs with train/validation in each epoch
  - Saves best model based on validation Top-1 accuracy
  - Periodic checkpoints every 20 epochs
  - Generates loss curves, accuracy plots, confusion matrices, and classification reports
  - Logs metrics to text files and JSON

### `data_setup.py`
Data loading and preprocessing:
- **create_dataloaders()**: Creates PyTorch DataLoaders
  - Uses ImageFolder structure for multi-class organization
  - Applies transforms: Resize to 224×224, TrivialAugmentWide augmentation, ToTensor normalization
  - Returns train, validation, and test loaders with class names

### `utils.py`
Comprehensive utility functions:
- **Metrics**: Top-k accuracy, precision, recall, F1-score, AUC (micro/macro/weighted)
- **Plotting**: Loss curves, accuracy curves, Top-1/Top-3 comparison
- **Confusion Matrix**: Heatmap visualization per dataset split
- **Classification Report**: Precision/recall/F1 per class
- **Model Management**: Save/load models, model summaries
- **File I/O**: Save metrics to text and JSON formats

### `prediction_for_one.py`
Inference script for single image predictions:
- Load a trained ViT model
- Make predictions on individual soybean leaf images
- Output predicted disease class and confidence scores

##  Quick Start

### Prerequisites
- Python 3.8+
- CUDA 10.1+ (for GPU acceleration, recommended)
- 40GB+ GPU memory (or adjust batch size for smaller GPUs)

### Installation

1. **Clone or download the project**

2. **Create and activate a virtual environment**
   ```bash
   # Using conda
   conda create -n soybean-vit python=3.9
   conda activate soybean-vit
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiment

```bash
python experiment.py
```

**What happens:**
1. Loads the SoyMulticlass dataset from specified directories
2. Creates train/validation/test DataLoaders with TrivialAugmentWide augmentation
3. Initializes the Vision Transformer with 12 attention heads and 12 transformer layers
4. Trains for 100 epochs with Adam optimizer (LR = 1e-5)
5. Saves the best model based on validation accuracy
6. Generates comprehensive evaluation metrics and visualizations

**Output Structure:**
```
Save_dir_100ep_dm768_lre5_ly12_hd12/
├── best_model.pth                      # Best model checkpoint
├── model_summary.txt                   # Model architecture summary
├── training_metrics.txt                # Epoch-wise metrics
├── detailed_metrics.txt                # Comprehensive final metrics
├── results.json                        # All results in JSON format
├── loss_curve.png                      # Training/validation loss plot
├── accuracy_curve.png                  # Training/validation accuracy plot
├── top1_top3_comparison.png           # Top-1 and Top-3 accuracy comparison
├── Train_confusion_matrix.png         # Training set confusion matrix
├── Test_confusion_matrix.png          # Test set confusion matrix
├── Train_classification_report.txt    # Per-class train metrics
├── Test_classification_report.txt     # Per-class test metrics
├── checkpoints/                        # Periodic model checkpoints
│   ├── model_epoch_20.pth
│   ├── model_epoch_40.pth
│   └── ...
└── *_model.pth                        # Final trained model
```

##  Configuration

### Modifying Hyperparameters

Edit `experiment.py` to customize training:

```python
# Training control
NUM_EPOCHS = 100              # Increase for more training
BATCH_SIZE = 32               # Reduce if GPU memory is limited

# Optimizer parameters
LEARNING_RATE = 0.00001       # Adam learning rate

# Vision Transformer architecture
D_MODEL = 768                 # Model dimensionality (embedding size)
N_HEADS = 12                  # Number of attention heads
N_LAYERS = 12                 #  KEY: Reduce this for better results on small datasets
IMAGE_SIZE = (224, 224)       # Input image dimensions
PATCH_SIZE = (16, 16)         # Patch size (determines number of patches)
```



###  Expected Performance

On the SoyMulticlass dataset with the recommended configuration:

| Metric | Expected Value |
|---|---|
| **Test Top-1 Accuracy** | ~85% |
| **Test Top-3 Accuracy** | ~98%+ |
| **Training Time** | ~2-3 hours (on A100 GPU) |
| **Per-Class Accuracy** | 80-90% (varies by disease type) |

##  Single Image Prediction

To make predictions on a single image:

```bash
python prediction_for_one.py --model_path path/to/best_model.pth
```

Edit these in `prediction_for_one.py`:
```python
IMG_PATH = "/path/to/soybean/leaf/image.jpg"
class_names = ["AB", "BP", "HL", "YMV"]  # Disease classes
N_LAYERS = 12  # Match your trained model's layer count
```

##  Vision Transformer Architecture Details

### Patch Embedding
Images are divided into 16×16 patches:
- 224×224 image ÷ 16×16 patches = 14×14 = **196 patches**
- Each patch is projected to 768-dimensional embeddings
- A learnable [CLS] token is prepended → **197 tokens total**

### Attention Mechanism
Multi-head self-attention with 12 heads:
- Each head attends to different feature representations
- Head dimension = 768 ÷ 12 = **64 dimensions per head**
- Allows the model to capture diverse spatial and semantic information

### Feed-Forward Network
Each transformer layer includes an MLP:
- Expansion: 768 → 3,072 (4× d_model)
- Activation: GELU
- Contraction: 3,072 → 768
- Dropout: 0.1 (helps prevent overfitting)

### Classification Head
Final classification on the [CLS] token:
- Input: [CLS] token (768 dimensions)
- Hidden layer: 768 → 3,072 (GELU activation)
- Output layer: 3,072 → 4 (classes: AB, BP, YMV, HL)
- Softmax normalization

## Key Research Insights

1. **Layer Reduction is Critical**: Reducing from 12 to fewer layers improved accuracy by 38% (47% → 85%)
2. **Patch Size Matters**: 16×16 patches provide good balance between patch count and feature specificity
3. **Positional Encoding**: Sinusoidal positional encodings work well for variable image patches
4. **Data Augmentation**: TrivialAugmentWide augmentation helps with generalization on small datasets
5. **Batch Normalization**: Layer normalization (not batch norm) is crucial in transformers

##  Customization

### Adding a New Disease Class

1. Update dataset folder structure:
   ```
   SoyMulticlass/
   ├── train/
   │   ├── AB/
   │   ├── BP/
   │   ├── YMV/
   │   ├── HL/
   │   └── NewDisease/  # Add here
   └── ...
   ```

2. Class names are automatically detected from directories via ImageFolder

### Changing Loss Function

In `experiment.py`:
```python
# Current: Cross-Entropy (good for multi-class)
loss_fn = torch.nn.CrossEntropyLoss()

# Alternative: Focal Loss (for imbalanced classes)
from torch.nn import BCEWithLogitsLoss
loss_fn = BCEWithLogitsLoss()
```

### Adjusting Image Size

For faster training/inference:
```python
IMAGE_SIZE = (128, 128)  # Smaller images = faster training
PATCH_SIZE = (8, 8)      # Adjust patches accordingly
```

##  Important Notes

- **Dataset Paths**: Update `train_dir`, `test_dir`, and `real_test_dir` in `experiment.py` to match your system
- **GPU Memory**: The default batch size 32 requires ~35GB GPU memory. Reduce if needed:
  ```python
  BATCH_SIZE = 16  # For 20GB GPU
  BATCH_SIZE = 8   # For 10GB GPU
  ```
- **Reproducibility**: Set random seeds for consistent results (add to `experiment.py`):
  ```python
  import random
  random.seed(42)
  torch.manual_seed(42)
  ```

##  References

- Dosovitskiy et al. (2020). "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (Original ViT paper)
- Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG reference)
- Kingma & Ba (2014). "Adam: A Method for Stochastic Optimization"
- TrivialAugment: https://arxiv.org/abs/2103.10158


