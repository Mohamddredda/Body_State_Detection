# Celeb-FBI Body States Detection

## 📌 Project Overview
This project is a computer vision and machine learning pipeline designed to predict physical and demographic attributes from facial images. Using the **Celeb-FBI Dataset**, the model classifies individuals into specific categories based on their age, height, weight, and gender.

The primary objective of this project is to demonstrate the impact of **Linear Discriminant Analysis (LDA)** on classification accuracy when applied to high-dimensional feature sets like **Histogram of Oriented Gradients (HOG)**.

## 🚀 Key Features
* **Automated Metadata Parsing:** Automatically extracts ground-truth labels (age, height, weight, gender) from image filenames.
* **HOG Feature Extraction:** Utilizes `skimage.feature.hog` to capture structural information and shape descriptors.
* **Dimensionality Reduction:** Implements LDA to reduce feature space while maximizing class separability.
* **Performance Benchmarking:** Provides detailed accuracy metrics comparing standard SVM vs. LDA-optimized SVM.
* **Interactive Inference:** Support for custom image uploads to test real-world attribute prediction.

## 📊 Performance Results
The integration of LDA significantly improved the model's ability to generalize across all physical attributes:

| Attribute | Accuracy (Before LDA) | Accuracy (After LDA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Age** | 80.15% | **95.56%** | +15.41% |
| **Height** | 62.32% | **91.67%** | +29.35% |
| **Weight** | 77.38% | **97.02%** | +19.64% |
| **Gender** | 85.98% | **97.85%** | +11.87% |

## 🛠️ Technologies Used
* **Python:** Language.
* **OpenCV (cv2):** Image reading and resizing.
* **Scikit-Image:** HOG feature extraction.
* **Scikit-Learn:** Machine learning (SVC, LDA, StandardScaler, train_test_split).
* **Matplotlib:** Performance visualization and image display.
* **Numpy:** Data manipulation and array processing.

## 📖 Pipeline Workflow
1.  **Preprocessing:** Images are converted to grayscale and resized to a uniform 150x150 resolution.
2.  **Binning:** Labels are discretized into categories:
    * **Age:** Young (<20), Adult (20-50), Old (>50).
    * **Height:** Short (<5.0), Average (5.0-5.8), Tall (>5.8).
    * **Weight:** Underweight (<50), Normal (50-100), Overweight (>100).
3.  **Feature Vectorization:** HOG descriptors are computed to form the primary dataset.
4.  **Scaling:** Features are normalized using `StandardScaler`.
5.  **Optimization:** LDA is applied to compress the features based on class labels.
6.  **Classification:** Final predictions are made using an RBF-kernel SVM.
