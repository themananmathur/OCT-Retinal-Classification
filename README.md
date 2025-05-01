## OCT Retinal Disease Classification Using Deep Learning

A high-performance pipeline for automated classification of retinal pathologies from Optical Coherence Tomography (OCT) scans using state-of-the-art deep learning architectures.

---

## Datasets

- **OCTID**  
  Multi-class classification (5 classes):  
  *Macular Hole, Age-Related Macular Degeneration, Central Serous Retinopathy, Diabetic Retinopathy, Normal*

- **OCTDL**  
  Binary classification:  
  *Normal vs Abnormal*

- **(Planned)** OCT5k  
  Multi-disease, multi-grade dataset for clinically realistic generalization tasks.

---

## Architectures

| Model              | Summary                                       |
|--------------------|-----------------------------------------------|
| DenseNet121        | Densely connected CNN for efficient feature reuse |
| MobileNetV2        | Lightweight architecture optimized for speed and accuracy |
| EfficientNetV2B2   | High-accuracy, compound-scaled CNN architecture |
| Swin Transformer   | Hierarchical Vision Transformer using shifted windows |

All models were trained using:
- 5-Fold Stratified Cross-Validation
- Class-balanced loss functions
- OneCycle learning rate scheduling
- Extensive image augmentations for generalization

---

## Results

| Dataset | Model            | Accuracy (%) |
|---------|------------------|--------------|
| OCTID   | EfficientNetV2B2 | ~90.3        |
| OCTDL   | EfficientNetV2B2 | ~92.7        |

- All metrics, confusion matrices, and classification reports are stored under `/logs/`.
- Best model checkpoints are automatically selected based on validation F1-score.

---

## Model Weights

All trained weights are hosted on Hugging Face:  
[https://huggingface.co/mananmathur16/OCT-Retinal-Classification](https://huggingface.co/mananmathur16/OCT-Retinal-Classification)

---

## Technology Stack

- Python 3.10  
- PyTorch, Hugging Face Transformers  
- Scikit-learn, OpenCV  
- Jupyter, Colab Pro, TensorBoard  
- Matplotlib, Seaborn

---

## Project Structure

```plaintext
OCT-Retinal-Classification/
├── models/          # Trained model weights
├── notebooks/       # Training notebooks per model and dataset
├── utils/           # Custom preprocessing, loaders, and evaluation utilities
├── logs/            # Training logs, classification reports, and visualizations
├── README.md        # Project overview
└── requirements.txt # All dependencies
```

## Citation

[1] P. Gholami and H. Rivaz, “Octid: Optical coherence tomography image database,” Computers and Electrical Engineering, vol. 81, p. 106522, 2020.

[2] Y. Costa et al., “Octdls: An open-access optical coherence tomography dataset for deep learning studies on retinal disease detection,” Scientific Data, vol. 11, no. 1, p. 182, 2024.
