# 🩻 Fracture Image Classification using Medical X-ray Images

## 📌 Project Overview
This project focuses on the automatic detection of bone fractures using real medical X-ray images through Machine Learning and Computer Vision techniques.  
It addresses a binary classification problem where the model predicts whether an X-ray image shows a fracture or not.

The project was developed in Google Colab and follows a structured Data Science workflow including data exploration, preprocessing, model training, and evaluation.

---

## 🎯 Objective
The main objective of this project is to:
- Classify X-ray images into fractured and non-fractured categories
- Develop a robust image classification model
- Apply Computer Vision techniques to a real medical dataset
- Evaluate model performance using appropriate classification metrics

---

## 🧠 Problem Type
- Task: Binary Classification  
- Domain: Medical Imaging / Computer Vision  
- Target Variable:
  - 0 → No Fracture
  - 1 → Fracture

---

## 📊 Dataset
- Type: Real medical X-ray images
- Format: Medical radiographs (image data)
- Use case: Automated fracture detection
- Source: *(Add dataset source here: Kaggle / Medical dataset / etc.)*

Note: Due to the size of medical image datasets, the full dataset is not included in this repository.

---

## 🔬 Methodology
The project follows an end-to-end Data Science and Machine Learning workflow:

1. Data loading and inspection  
2. Image preprocessing (resizing, normalization)  
3. Exploratory Data Analysis (EDA)  
4. Model development for image classification  
5. Model training and validation  
6. Performance evaluation and analysis  

---

## 🔎 Exploratory Data Analysis (EDA)
During the EDA phase:
- Class distribution (fracture vs non-fracture) was analyzed
- Dataset balance and potential imbalance were evaluated
- Image variability and quality were inspected
- Preprocessing requirements were identified

This step was essential to ensure proper model training and performance.

---

## 🤖 Model Development
The model was trained to classify X-ray images using Machine Learning / Deep Learning techniques.

Key steps included:
- Image preprocessing and normalization
- Feature extraction
- Model training on labeled X-ray images
- Validation using unseen data

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score

---

## 🛠️ Technologies & Tools
- Python
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras *(edit if used)*
- OpenCV *(if used)*

---

## 📈 Results & Insights
- The model is capable of detecting fractures from medical X-ray images.
- Data preprocessing and image normalization significantly impacted model performance.
- Class balance analysis was crucial for reliable evaluation.
- The project demonstrates the potential of AI in medical image analysis and computer-aided diagnosis.

---

## 🚀 How to Run the Project
This project was developed in Google Colab.

Steps to reproduce:
1. Open the notebook in Google Colab
2. Upload the dataset or connect Google Drive
3. Install required libraries (if needed)
4. Run all cells sequentially



## 📁 Repository Structure

```
fracture-classification/
│
├── notebooks/        # Notebooks de Colab (EDA, modelado)
├── data/             # Dataset 
├── images/           # Ejemplos de rayos X 
├── README.md         # Documentación
└── requirements.txt  # Dependencias (opcional)
```

## ⚠️ Note on Medical Data

This project uses medical imaging data for educational and research purposes only.
It is not intended for clinical or diagnostic use.

## 👩‍💻 Author

Cristina

