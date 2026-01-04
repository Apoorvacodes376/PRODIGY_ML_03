# PRODIGY_ML_03



#  Cat vs Dog Image Classification using SVM

##  Project Overview
This project implements a **classical Machine Learning approach** to classify images of cats and dogs using the **Kaggle Dogs vs Cats dataset**.  
Instead of deep learning models, the project focuses on **feature extraction and Support Vector Machines (SVM)** to understand the fundamentals of image-based ML pipelines.

The project was completed as part of a **Machine Learning Internship at Prodigy InfoTech**.

---

##  Objective
- To build an end-to-end ML pipeline for image classification  
- To apply **HOG (Histogram of Oriented Gradients)** for feature extraction  
- To train and evaluate a **Support Vector Machine (SVM)** classifier  
- To analyze model performance using standard evaluation metrics  

---

##  Methodology
1. **Image Preprocessing**
   - Images resized to a fixed resolution
   - Converted to grayscale to reduce computational complexity

2. **Feature Extraction**
   - HOG features extracted to capture shape and edge information

3. **Model Training**
   - Linear Support Vector Machine (LinearSVC) used for classification
   - Feature scaling applied using `StandardScaler`

4. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score

---

##  Tech Stack
- **Programming Language:** Python  
- **Libraries Used:**
  - OpenCV
  - scikit-image
  - scikit-learn
  - NumPy
  - tqdm

---

##  Dataset
- **Source:** Kaggle – Dogs vs Cats Dataset  
- **Classes:** Cat, Dog  
- **Total Images Used:** 25,000  
- Dataset is automatically available in Kaggle notebooks under:
  ```
  /kaggle/input/dogs-vs-cats/train
  ```

---

##  Project Structure
```
├── svm_cat_dog.py            # Main training script
├── svm_cat_dog_full_model.pkl # Saved trained model
├── README.md                 # Project documentation
```

---

##  How to Run the Project

### 1️ Install Dependencies
```bash
pip install numpy opencv-python scikit-image scikit-learn tqdm
```

### 2️ Set Dataset Path
```python
DATASET_PATH = "/kaggle/input/dogs-vs-cats/train"
```

### 3️ Run the Script
```bash
python svm_cat_dog.py
```

---

##  Results
- **Accuracy:** ~72%
- **Balanced performance** across both classes
- Comparable results to other classical ML approaches on this dataset

```
Precision | Recall | F1-score
Cat: 0.73 | 0.70 | 0.72
Dog: 0.71 | 0.74 | 0.73
```

---

##  Observations
- Linear SVM provides faster training compared to kernel-based SVMs
- HOG features effectively capture structural patterns in images
- Classical ML models offer lower computational cost compared to CNNs

---

##  Future Improvements
- Hyperparameter tuning for SVM
- Experimenting with different feature descriptors
- Comparison with CNN-based approaches
- Confusion matrix visualization

---

##  Conclusion
This project demonstrates that **classical machine learning techniques**, when combined with effective feature extraction methods like HOG, can achieve reliable performance on image classification tasks. It provides a strong foundation for understanding computer vision concepts before transitioning to deep learning models.

---

##  Acknowledgements
- **Prodigy InfoTech** – for the internship opportunity  
- **Kaggle** – for providing the dataset  
- scikit-learn & OpenCV community  

---

