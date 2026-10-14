# Food Recognition Challenge (2nd Place) - Multi-Label Classification

This repository contains the code and methodology that secured 2nd place in the Advanced Machine Learning Food Recognition Challenge . The objective was to train deep learning models to predict the presence of 498 distinct food categories in color images within a multi-label classification environment. The dataset consisted of 39,962 training images and 1,000 test images sourced from the AICrowd Food Recognition Benchmark. Submissions were evaluated based on the Micro F1 score.

## 1. Repository Structure
* `Final_EDA_Notebook.ipynb`: Performs comprehensive exploratory data analysis, including label matrix distribution, multi-label characteristics, image dimension extraction, and dataset-specific normalization metrics computation.
* `Final_Individual_Notebook.ipynb`: The training pipeline for individual models, handling data augmentation, progressive layer unfreezing, threshold optimization, and standalone inference.
* `Final_Ensemble_Notebook.ipynb`: The weighted ensembling logic combining multiple model architectures, implementing fallback mechanisms, and generating the final submission.
* `AML Report.pdf`: A detailed written report explaining the methodology, experiments, results, and lessons learned.

## 2. Methodology & Pipeline

### Data Preprocessing & Augmentation
To prevent overfitting and improve generalization across the 498 classes, a robust augmentation pipeline was applied during training. 
* Training images underwent transformations including Random Horizontal/Vertical Flips, Random Rotation, Random Resized Crop, Color Jitter, Random Perspective, Random Affine transformations, Gaussian Blur, and Random Autocontrast. 
* Both training and testing images were resized and normalized using custom per-channel mean and standard deviation values derived from the training set.

### Model Architecture & Training
The solution leveraged a combination of advanced transformer architectures loaded via the `timm` library, specifically Vision Transformers (`vit_base_mci_224.apple_mclip`), SwinV1 (`swin_base_patch4_window7_224`), and SwinV2 (`swinv2_base_window12to16_192to256.ms_in22k_ft_in1k`). 
* **Progressive Unfreezing:** Early layers were frozen to retain pretrained features, while deeper blocks and the classification head were unfrozen for task-specific fine-tuning.
* **Optimization:** Models were trained across multiple GPUs using `nn.DataParallel`. The optimization stack utilized Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`), the AdamW optimizer with layer-specific learning rates, and a `StepLR` scheduler.
* **Dynamic Threshold Optimization:** Instead of using a static 0.5 threshold, an automated function swept a range of thresholds at the end of each epoch to find the optimal cut-off that maximized the Micro F1 score on the validation set.

### Ensemble Strategy & Inference
The final submission utilized a robust weighted majority voting ensemble to capitalize on the strengths of different architectures.
* **Weighted Voting:** Predictions from multiple SwinV1, SwinV2, and ViT models were combined using normalized weights derived from their individual public leaderboard scores.
* **Fallback Logic:** To address edge cases where all predicted probabilities for a test image fell below the optimal threshold, a fallback mechanism was implemented to assign the class with the highest probability. This ensured every test image received at least one label.
* **Influence Analysis:** The ensemble script includes diagnostic tools to track each model's agreement percentage and decisive voting power, calculating how often removing a specific model flipped the final ensemble prediction.

## 3. Technology Stack
* **Deep Learning:** PyTorch, Torchvision, `timm` (PyTorch Image Models).
* **Data Processing & Analysis:** NumPy, Pandas, Scikit-learn, Pillow (PIL).
* **Visualization & Utilities:** Matplotlib, Seaborn, tqdm, tabulate.
