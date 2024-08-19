# Predicting Classes of Inflammatory Bowel Disease using Computer Vision Classification

## Project Overview
This project aims to predict classes of Inflammatory Bowel Disease (IBD) using advanced computer vision techniques. It also analyzes the discrepancy between endoscopic and histology scores in assessing inflammation. The project leverages state-of-the-art pretrained models and machine learning techniques for accurate classification and regression analysis.

## Key Features
1. Computer vision-based classification of IBD
2. Regression analysis for predicting IBD severity scores
3. Analysis of discrepancies between endoscopic and histology scores
4. Utilization of advanced pretrained models:
   - SEResNet50
   - VisionMAE
   - DINOv2
5. Implementation of data augmentation techniques
6. Cross-validation for robust model evaluation

## Methodology

### Data Preparation
- Collection of endoscopic images and corresponding histology data
- Data preprocessing and cleaning
- Implementation of data augmentation techniques to enhance model generalization

### Model Development
- Utilization of pretrained models for feature extraction:
  - SEResNet50: Squeeze-and-Excitation ResNet50 for enhanced feature representation
  - VisionMAE: Vision Masked Autoencoder for self-supervised learning
  - DINOv2: Self-supervised vision transformer for robust feature extraction
- Fine-tuning of models for IBD classification and severity score prediction
- Implementation of cross-validation to ensure model robustness

### Classification Analysis
- Prediction of IBD classes using fine-tuned pretrained models
- Evaluation of model performance using metrics such as accuracy, precision, recall, and F1-score

### Regression Analysis
- Prediction of IBD severity scores using regression techniques
- Evaluation using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)

### Discrepancy Analysis
- Comparison of endoscopic predictions with histology scores
- Statistical analysis of discrepancies
- Identification of factors contributing to differences between visual and histological assessments

## Technologies Used
- Python
- PyTorch for deep learning models and pretrained architectures
- OpenCV for image processing
- Scikit-learn for model evaluation and cross-validation
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for data visualization


## Future Work
- Exploration of ensemble methods combining multiple pretrained models
- Investigation of more recent architectures for improved performance
- Development of a user-friendly interface for clinical use
- Integration of additional clinical data for more comprehensive analysis

This project demonstrates the application of advanced computer vision techniques in medical diagnosis, specifically for Inflammatory Bowel Disease. It highlights the potential of machine learning in assisting medical professionals, understanding the complexities of disease assessment, and bridging the gap between endoscopic and histological evaluations.
