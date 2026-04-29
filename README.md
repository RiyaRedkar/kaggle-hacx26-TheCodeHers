# 🌍 EuroSAT Land Classification System

### KaggleHacX ’26 — Data Sprint to the Peak

🚀 **An end-to-end deep learning solution for satellite image classification with real-time prediction and explainable outputs.**

---

## 🧠 Overview

This project presents a complete machine learning pipeline to classify satellite images into **10 land-use categories** using deep learning.

Built under strict hackathon constraints, the solution focuses on:

* High accuracy ⚡
* Clean reproducible pipeline 🔧
* Real-world usability 🌱

---

## 🎯 Problem Statement

Satellite imagery plays a crucial role in:

* Land use monitoring
* Agricultural planning
* Urban development

The goal is to **automatically classify satellite images** into predefined land categories using AI.

---

## 🧠 Model Architecture

We used:

👉 **ResNet18 (Pretrained on ImageNet)**
👉 **Transfer Learning Approach**

### Why this works:

* Efficient and fast training
* Strong feature extraction
* Performs well on limited time/data

---

## ⚙️ Methodology

### 🔹 Data Preprocessing

* Resized images to **224×224**
* Applied normalization using ImageNet statistics
* Applied augmentation:

  * Horizontal Flip
  * Rotation

---

### 🔹 Training Strategy

* 80-20 Train/Validation split
* Optimizer: Adam
* Loss: CrossEntropy
* Epochs: 3

---

### 🔹 Validation Performance

📈 **~94% Accuracy**

The model demonstrates strong generalization across all classes.

---

## 📊 Features

✔ Real-time image classification
✔ Confidence score visualization
✔ Top-3 prediction insights
✔ Clean and interactive UI
✔ Fully reproducible pipeline

---

## 🖥️ Demo Application

Built using **Streamlit**, the app allows users to:

1. Upload a satellite image
2. Get instant predictions
3. View confidence score and alternatives

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch torchvision streamlit
```

### 2. Run the app

```bash
streamlit run main/app.py
```

---

## 📁 Project Structure

```
kaggle-hacx26/
│
├── main/
│   ├── train.py
│   ├── app.py
│   └── model.pth
│
├── notebooks/
│   └── eurosat_training.ipynb
│
└── submission/
    ├── solution_teamname.csv
    ├── methodology.pdf
    └── demo_video.mp4
```

---

## 🧪 Key Insight

We ensured **consistency between training and inference preprocessing**, which significantly improved prediction reliability.

---

## 🌍 Real-World Impact

This system can be extended for:

* Satellite-based land monitoring
* Environmental analysis
* Smart agriculture solutions

---

## 👩‍💻 Team

**TheCodeHers** 💡

---

## 🏁 Conclusion

This project demonstrates how a well-structured approach combining:

* Transfer learning
* Clean engineering
* Thoughtful UI

can deliver a powerful and practical AI solution within a short time.

---

⭐ *Built with speed, precision, and purpose.*
