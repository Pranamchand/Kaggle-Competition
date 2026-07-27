<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:2c5364,100:00c6ff&height=220&section=header&text=Kaggle%20Competitions&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Two%20End-to-End%20Machine%20Learning%20Models&descAlignY=58&descAlign=50" width="100%"/>

<a href="https://github.com/Pranamchand">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=800&color=00C6FF&center=true&vCenter=true&width=650&lines=Real-World+Tabular+%2B+NLP+Classification;Disaster+Tweets+%7C+Spaceship+Titanic;Full+EDA+%E2%86%92+Feature+Engineering+%E2%86%92+Modeling+%E2%86%92+Submission" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-1E4620?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

</div>

<br/>

## 📌 Repository Overview

This repository is a collection of three **Kaggle "Getting Started" competition** solutions, each built as a complete, self-contained ML pipeline — from raw data to a submitted `submission.csv`. Both notebooks follow a consistent, disciplined workflow: explore the data first, engineer features that actually matter, benchmark multiple models honestly, and only then tune the winner.

<div align="center">

| # | Project | Task Type | Kaggle Leaderboard Rank |
|:-:|:--|:--|:-:|
| 1 | 🌪️ [Disaster Tweets Classification](#-1-disaster-tweets-classification) | Binary NLP Classification | 🏅 **628** |
| 2 | 🚀 [Spaceship Titanic Prediction](#-2-spaceship-titanic-prediction) | Binary Tabular Classification | 🏅 **1733** |
| 3 | 🔢 [Digit Recognizer (MNIST)](#-3-digit-recognizer-mnist-classification) | Multiclass Image Classification (CNN) | 🏅 **667** |

</div>

<div align="center">

`🧮 58,306 training records analyzed`  •  `🤖 10 models benchmarked`  •  `🔬 2 GridSearchCV tuning runs`  •  `🎯 3 leaderboard submissions`

</div>

<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=3&width=100%"/>

<br/>

## 🌪️ 1. Disaster Tweets Classification

<div align="center">
<img src="https://img.shields.io/badge/Kaggle%20Rank-%23628-success?style=flat-square&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Best%20Model-Logistic%20Regression-blue?style=flat-square"/>
<img src="https://img.shields.io/badge/Accuracy-81.55%25-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/F1--Score-0.7672-yellow?style=flat-square"/>
</div>

**Goal:** Given a tweet, predict whether it is describing a real disaster (`target = 1`) or not (`target = 0`) — a classic NLP text-classification problem on the [Kaggle "Natural Language Processing with Disaster Tweets"](https://www.kaggle.com/competitions/nlp-getting-started) dataset (7,613 training tweets, 3,263 test tweets).


<br/>

## 🚀 2. Spaceship Titanic Prediction

<div align="center">
<img src="https://img.shields.io/badge/Kaggle%20Rank-%231733-success?style=flat-square&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Best%20Model-XGBoost-006400?style=flat-square"/>
<img src="https://img.shields.io/badge/Accuracy-80.97%25-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/F1--Score-0.8114-yellow?style=flat-square"/>
</div>

**Goal:** Predict whether a passenger aboard the *Spaceship Titanic* was transported to an alternate dimension after the ship's collision with a spacetime anomaly — a tabular binary classification problem on the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) dataset (8,693 training passengers).


<br/>

## 🔢 3. Digit Recognizer (MNIST) Classification

<div align="center">
<img src="https://img.shields.io/badge/Best%20Model-CNN%20(Keras)-blue?style=flat-square"/>
<img src="https://img.shields.io/badge/Test%20Accuracy-98.60%25-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/Val%20Loss-0.041-yellow?style=flat-square"/>
</div>

**Goal:** Given a 28×28 grayscale image of a handwritten digit (flattened into 784 pixel columns), predict which digit (0–9) it represents — a classic multiclass image-classification problem on the [Kaggle "Digit Recognizer"](https://www.kaggle.com/competitions/digit-recognizer) dataset (42,000 training images, 28,000 test images).

**Approach:** Pixel values were scaled to a 0–1 range and reshaped from flat 784-length vectors into 28×28×1 image tensors, with labels one-hot encoded across 10 classes. A **Convolutional Neural Network** built in Keras — two `Conv2D` + `MaxPooling2D` blocks followed by a dense layer with dropout for regularization and a softmax output layer — was trained for 5 epochs, reaching **98.60% accuracy** on the held-out validation split.


<br/>

*"Every leaderboard rank starts with a messy `train.csv` and a stubborn refusal to skip the EDA."* 🚀

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c6ff,50:2c5364,100:0f2027&height=120&section=footer" width="100%"/>

</div>