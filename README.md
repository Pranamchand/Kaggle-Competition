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

</div>

<br/>

## 📌 Repository Overview

This repository is a collection of two **Kaggle "Getting Started" competition** solutions, each built as a complete, self-contained ML pipeline — from raw data to a submitted `submission.csv`. Both notebooks follow a consistent, disciplined workflow: explore the data first, engineer features that actually matter, benchmark multiple models honestly, and only then tune the winner.

<div align="center">

| # | Project | Task Type | Kaggle Leaderboard Rank |
|:-:|:--|:--|:-:|
| 1 | 🌪️ [Disaster Tweets Classification](#-1-disaster-tweets-classification) | Binary NLP Classification | 🏅 **628** |
| 2 | 🚀 [Spaceship Titanic Prediction](#-2-spaceship-titanic-prediction) | Binary Tabular Classification | 🏅 **1733** |

</div>

<div align="center">

`🧮 16,306 training records analyzed`  •  `🤖 9 models benchmarked`  •  `🔬 2 GridSearchCV tuning runs`  •  `🎯 2 leaderboard submissions`

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

### 🔁 Notebook Workflow

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "26px", "primaryTextColor": "#ffffff", "lineColor": "#00c6ff"}, "flowchart": {"nodeSpacing": 90, "rankSpacing": 110, "curve": "basis"}} }%%
flowchart TD
    A["📥&nbsp;&nbsp;Load Data"] --> B["🔎&nbsp;&nbsp;EDA"]
    B --> C["🧹&nbsp;&nbsp;Preprocessing"]
    C --> D["⚙️&nbsp;&nbsp;Feature Engineering"]
    D --> E["🤖&nbsp;&nbsp;Model Training & Evaluation"]
    E --> F["🎯&nbsp;&nbsp;Trial & Error Tuning"]
    F --> G["📄&nbsp;&nbsp;Generate submission.csv"]
    G --> H["🚀&nbsp;&nbsp;Submit to Kaggle"]

    style A fill:#0f2027,color:#fff,stroke:#00c6ff,stroke-width:2px
    style B fill:#173544,color:#fff,stroke:#00c6ff,stroke-width:2px
    style C fill:#204a5c,color:#fff,stroke:#00c6ff,stroke-width:2px
    style D fill:#00697a,color:#fff,stroke:#00c6ff,stroke-width:2px
    style E fill:#00909e,color:#fff,stroke:#00c6ff,stroke-width:2px
    style F fill:#00b4bd,color:#000,stroke:#00c6ff,stroke-width:2px
    style G fill:#4dd8ff,color:#000,stroke:#00c6ff,stroke-width:2px
    style H fill:#9fecff,color:#000,stroke:#00c6ff,stroke-width:2px
```


<br/>

## 🚀 2. Spaceship Titanic Prediction

<div align="center">
<img src="https://img.shields.io/badge/Kaggle%20Rank-%231733-success?style=flat-square&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Best%20Model-XGBoost-006400?style=flat-square"/>
<img src="https://img.shields.io/badge/Accuracy-80.97%25-orange?style=flat-square"/>
<img src="https://img.shields.io/badge/F1--Score-0.8114-yellow?style=flat-square"/>
</div>

**Goal:** Predict whether a passenger aboard the *Spaceship Titanic* was transported to an alternate dimension after the ship's collision with a spacetime anomaly — a tabular binary classification problem on the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) dataset (8,693 training passengers).

### 🔁 Notebook Workflow

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "26px", "primaryTextColor": "#ffffff", "lineColor": "#9933ff"}, "flowchart": {"nodeSpacing": 90, "rankSpacing": 110, "curve": "basis"}} }%%
flowchart TD
    A["📥&nbsp;&nbsp;Load Data"] --> B["🔎&nbsp;&nbsp;EDA"]
    B --> C["🧹&nbsp;&nbsp;Data Preprocessing"]
    C --> D["⚙️&nbsp;&nbsp;Feature Engineering"]
    D --> E["🤖&nbsp;&nbsp;Model Selection"]
    E --> F["🎯&nbsp;&nbsp;Model Tuning"]
    F --> G["📄&nbsp;&nbsp;Generate submission.csv"]

    style A fill:#1a0033,color:#fff,stroke:#bf80ff,stroke-width:2px
    style B fill:#2b0052,color:#fff,stroke:#bf80ff,stroke-width:2px
    style C fill:#40007a,color:#fff,stroke:#bf80ff,stroke-width:2px
    style D fill:#5c00b3,color:#fff,stroke:#bf80ff,stroke-width:2px
    style E fill:#7a1aff,color:#fff,stroke:#bf80ff,stroke-width:2px
    style F fill:#9933ff,color:#fff,stroke:#bf80ff,stroke-width:2px
    style G fill:#cc99ff,color:#000,stroke:#bf80ff,stroke-width:2px
```


<br/>

*"Every leaderboard rank starts with a messy `train.csv` and a stubborn refusal to skip the EDA."* 🚀

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c6ff,50:2c5364,100:0f2027&height=120&section=footer" width="100%"/>

</div>