# Practical Lab 3 - Vanilla CNN and Fine-Tune VGG16
**Author:** Ali Cihan Ozdemir (Student ID: 9091405)

## 📌 Project Overview
This project focuses on resolving a classic binary image classification problem: distinguishing between images of **Dogs and Cats**.

### 📁 Dataset Source
As officially requested, this lab utilizes the **[Official Microsoft Kaggle Cats and Dogs Dataset](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)** provided by Microsoft Research. To ensure computational efficiency and objective analysis, the working dataset is deliberately isolated as a symmetrically balanced **5,000-image subset** (2,500 dogs and 2,500 cats) algorithmically extracted from the original 25,000 image pool.

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

### Execution (Best Practice: Virtual Environment)
To prevent unexpected dependency conflicts and adhere to modern best practices, we highly recommend executing this project within a clean Python virtual environment.

1. **Initialize the virtual environment:**
   ```bash
   python3 -m venv venv
   ```
2. **Activate the environment:**
   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
3. **Install the required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Launch the fully self-contained notebook:**
   ```bash
   jupyter notebook Lab3_Dogs_vs_Cats.ipynb
   ```

*⚠️ **Note:** The notebook operates totally autonomously! Processing the notebook explicitly top-to-bottom will trigger the programmatic downloading, extraction, and precise 5,000-file subset isolation of the working dataset, followed immediately by standard model training and localized `.h5` weight generation.*
