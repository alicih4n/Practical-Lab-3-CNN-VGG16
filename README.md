# Practical Lab 3 - Vanilla CNN and Fine-Tune VGG16
**Author:** Ali Cihan Ozdemir (Student ID: 9091405)

## 📌 Project Overview
This project focuses on resolving a classic binary image classification problem: distinguishing between images of **Dogs and Cats**. To ensure computational efficiency and objective analysis, the dataset utilized is a symmetrically balanced 5,000-image subset extracted from the core corpus (2,500 dogs and 2,500 cats).

## 🔬 Methodology
In this laboratory, two distinct Deep Learning models are architected, trained, and comparatively evaluated:

1. **Custom Vanilla CNN**
   - A convolutional neural network built entirely from scratch. 
   - Architected using stacked convolutional and max-pooling blocks, capped with a dense grouping incorporating heavy dropout for native regularization.

2. **Fine-Tuned VGG16 (Transfer Learning)**
   - A state-of-the-art model natively pre-trained on the massive ImageNet dataset. 
   - Developed by freezing the early feature-extraction baseline layers to preserve pre-learned structural knowledge, then dynamically fine-tuning an uncoupled binary classification head.

*During the model optimization phase, strict **`ModelCheckpoint`** callbacks were utilized to sequentially register and capture the most optimally generalized model weights (yielding locally as `.h5` files).*

## 📊 Evaluation Metrics
To comprehensively validate mapping behavior and class predictability beyond trivial binary accuracy, the final trained architectures were subjected to rigorous evaluation metrics against a locked test distribution:
- **Overall Macro Accuracy & Test Loss Verification**
- **Confusion Matrices**: Built via Seaborn heatmaps to visibly quantify spatial false negatives vs false positives.
- **Classification Reports**: Extracting critical F1-Scores, Precision, and Recall ranges.
- **Precision-Recall Curves**: Graphing Area Under the Curve (PR AUC) to formally judge mapping boundaries.
- **Misclassification Analysis**: Dynamically parsing error tensors to project a visual grid isolating the precise raw images where the network failed, paired with true/predicted red-flags.

## ⚙️ Repository Structure & How to Run
This repository is engineered to be exceptionally minimal and strictly academic. Large raw datasets, mapped image extractions, localized weight caches, and environment directories are strictly excluded via the provided `.gitignore`. 

### Execution
1. Install the required libraries locally:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the fully self-contained notebook:
   ```bash
   jupyter notebook Lab3_Dogs_vs_Cats.ipynb
   ```

*⚠️ **Note:** The notebook operates totally autonomously! Processing the notebook explicitly top-to-bottom will trigger the programmatic downloading, extraction, and precise 5,000-file subset isolation of the working dataset, followed immediately by standard model training and localized `.h5` weight generation.*
