# OCT Retinal Classification using Deep Learning

This project focuses on the automated classification of retinal abnormalities from Optical Coherence Tomography (OCT) images using state-of-the-art deep learning architectures.

# Datasets Used

- OCTID: Multi-class dataset with 5 retinal classes  
- OCTDL: Binary dataset with Normal and Abnormal classification  
- (Future Work) OCT5k: Multi-disease and multi-graded dataset

# Models Trained

- `DenseNet121`
- `MobileNetV2`
- `EfficientNetV2B2`
- `Swin Transformer`

> All models trained with 5-Fold Stratified Cross-Validation, class weighting, learning rate schedulers, and extensive augmentation.

# Results

- EfficientNetV2B2 (OCTID): ~90.3% Accuracy  
- EfficientNetV2B2 (OCTDL): ~92.7% Accuracy  
- Confusion Matrices and Classification Reports saved in `/logs/`

# Model Weights

All trained model weights hosted on Hugging Face:
[https://huggingface.co/mananmathur16/OCT-Retinal-Classification](https://huggingface.co/mananmathur16/OCT-Retinal-Classification)

# Technologies

- Python, PyTorch, Transformers (Hugging Face)
- Jupyter, Google Colab, TensorBoard
- Matplotlib, Seaborn, Sklearn

# Author

Manan Mathur  
B.Tech IT, MIT Manipal  
Midterm Project (2025)

---

# Folder Structure
OCT-Retinal-Classification/
├── models/ # Final saved weights (.pth, .keras)
├── notebooks/ # All training notebooks
├── utils/ # Utility scripts
├── logs/ # Training logs, classification reports
├── README.md # This file
└── requirements.txt # Dependencies


# Citation

[1] P. Gholami and H. Rivaz, “Octid: Optical coherence tomography image database,” Computers and Electrical Engineering, vol. 81, p. 106522, 2020.

[2] Y. Costa et al., “Octdls: An open-access optical coherence tomography dataset for deep learning studies on retinal disease detection,” Scientific Data, vol. 11, no. 1, p. 182, 2024.
