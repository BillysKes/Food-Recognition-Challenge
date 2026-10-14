# Food Recognition Challenge (2nd Place) - Multi-Label Classification

This repository contains the code and methodology that secured 2nd place in the Advanced Machine Learning Food Recognition Challenge[cite: 95]. The objective was to train deep learning models to predict the presence of 498 distinct food categories in color images within a multi-label classification environment[cite: 95]. The dataset consisted of 39,962 training images and 1,000 test images sourced from the AICrowd Food Recognition Benchmark[cite: 95]. Submissions were evaluated based on the Micro F1 score[cite: 95].

## 1. Repository Structure
* `Final_EDA_Notebook.ipynb`: Performs comprehensive exploratory data analysis, including label matrix distribution, multi-label characteristics, image dimension extraction, and dataset-specific normalization metrics computation[cite: 95].
* `Final_Individual_Notebook.ipynb`: The training pipeline for individual models, handling data augmentation, progressive layer unfreezing, threshold optimization, and standalone inference[cite: 95].
* `Final_Ensemble_Notebook.ipynb`: The weighted ensembling logic combining multiple model architectures, implementing fallback mechanisms, and generating the final submission[cite: 95].
* `AML Report.pdf`: A detailed written report explaining the methodology, experiments, results, and lessons learned[cite: 95].

## 2. Methodology & Pipeline

### Data Preprocessing & Augmentation
To prevent overfitting and improve generalization across the 498 classes, a robust augmentation pipeline was applied during training[cite: 95]. 
* Training images underwent transformations including Random Horizontal/Vertical Flips, Random Rotation, Random Resized Crop, Color Jitter, Random Perspective, Random Affine transformations, Gaussian Blur, and Random Autocontrast[cite: 95]. 
* Both training and testing images were resized and normalized using custom per-channel mean and standard deviation values derived from the training set[cite: 95].

### Model Architecture & Training
The solution leveraged a combination of advanced transformer architectures loaded via the `timm` library, specifically Vision Transformers (`vit_base_mci_224.apple_mclip`), SwinV1 (`swin_base_patch4_window7_224`), and SwinV2 (`swinv2_base_window12to16_192to256.ms_in22k_ft_in1k`)[cite: 95]. 
* **Progressive Unfreezing:** Early layers were frozen to retain pretrained features, while deeper blocks and the classification head were unfrozen for task-specific fine-tuning[cite: 95].
* **Optimization:** Models were trained across multiple GPUs using `nn.DataParallel`[cite: 95]. The optimization stack utilized Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`), the AdamW optimizer with layer-specific learning rates, and a `StepLR` scheduler[cite: 95].
* **Dynamic Threshold Optimization:** Instead of using a static 0.5 threshold, an automated function swept a range of thresholds at the end of each epoch to find the optimal cut-off that maximized the Micro F1 score on the validation set[cite: 95].

### Ensemble Strategy & Inference
The final submission utilized a robust weighted majority voting ensemble to capitalize on the strengths of different architectures[cite: 95].
* **Weighted Voting:** Predictions from multiple SwinV1, SwinV2, and ViT models were combined using normalized weights derived from their individual public leaderboard scores[cite: 95].
* **Fallback Logic:** To address edge cases where all predicted probabilities for a test image fell below the optimal threshold, a fallback mechanism was implemented to assign the class with the highest probability[cite: 95]. This ensured every test image received at least one label[cite: 95].
* **Influence Analysis:** The ensemble script includes diagnostic tools to track each model's agreement percentage and decisive voting power, calculating how often removing a specific model flipped the final ensemble prediction[cite: 95].

## 3. Technology Stack
* **Deep Learning:** PyTorch, Torchvision, `timm` (PyTorch Image Models)[cite: 95].
* **Data Processing & Analysis:** NumPy, Pandas, Scikit-learn, Pillow (PIL)[cite: 95].
* **Visualization & Utilities:** Matplotlib, Seaborn, tqdm, tabulate[cite: 95].
