This is the machine learning project for my mendelu class. The Dataset wasn't included because the files were too large. 

Dataset: https://www.kaggle.com/datasets/datasnaek/chess


Chess Win Predictor (The "Detect-o-meter")

![alt text](https://img.shields.io/badge/Python-3.10+-blue.svg)
![alt text](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![alt text](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)

A Machine Learning pipeline that predicts the winner of a chess game (White/Black/Draw) based on a single static board snapshot. Unlike traditional engines (Stockfish) that use Minimax search to calculate future moves, this project uses pattern recognition to evaluate the current position immediately.

# 🎯 Project Goals

Static Evaluation: Predict game outcomes without looking ahead (Search Depth = 0).

Comparative Analysis: Benchmark traditional ML (LogReg, RF, KNN) against Deep Learning (MLP).

Visualization: Create a real-time "Win Probability" video overlay for PGN game replays.

# 📊 Key Findings

We analyzed over 1.2 million board states derived from 20,000 amateur Lichess games.

Model	Architecture	Accuracy	Key Insight
Logistic Regression	Linear	66.1%	Highly stable. Proved Material + Mobility = Win is the dominant linear factor.
Random Forest	Ensemble Tree	66.7%	The best "Traditional" model. Hit a ceiling due to inability to model complex sacrifices.
KNN	Geometric	63.1%	Required PCA (Dimensionality Reduction) to function. Struggled with high-dimensional board data.
Deep Learning	MLP (Neural Net)	71.0%	The Breakdown. Successfully learned non-linear Feature Synthesis (e.g., Compensation for sacrificed material).

The "Static Wall": Traditional models plateaued at ~66%. The MLP breaking 70% suggests that Deep Learning can identify positional nuances that rule-based models miss, though the remaining error rate is largely attributed to unpredictable human blunders (tactics) inherent in amateur play.

data/raw/games.csv            # Original Lichess Dataset

data/processed/master_data.csv # Generated dataset (1.2M rows)

models/                      # Saved .pkl (Sklearn) and .keras (TensorFlow) models

plots/                       # Generated confusion matrices & feature importance charts

src/

   01_preprocessing.py             # Parses PGNs, simulates games, extracts features
   
   02_train_rf_enhanced.py         # Random Forest training
   
   03_train_logreg_enhanced.py     # Logistic Regression training
   
   04_train_knn_enhanced.py        # KNN training (with PCA)
   
   05_train_dl_enhanced.py         # Deep Learning (MLP) training
   
   07_make_video_mp4.py            # Generates the "Detect-o-meter" video


# 🚀 Usage Pipeline

Run the scripts in the following order to replicate the results.

1. Data Processing

Extracts features from raw PGN strings.

Input: data/raw/games.csv

Key Engineering: Calculates Material Difference and Mobility Difference (using a null-move lookahead to count opponent legal moves).

python src/01_preprocessing.py

2. Model Training

Train the individual models. Each script saves the model to /models and generates analysis charts in /plots.

Baseline Linear Model
python src/03_train_logreg_enhanced.py

Ensemble Tree Model
python src/02_train_rf_enhanced.py

Geometric Model (Includes PCA Step)
python src/04_train_knn_enhanced.py

Deep Learning Model (The best performer)
python src/06_train_deep_learning_enhanced.py
3. Generate Visualization

Runs the "Detect-o-meter" on a sample game (Deep Blue vs. Kasparov, 1997) to visualize how the models "think" over time.
python src/07_make_video_mp4.py

#  🧠 Methodology Highlights
Feature Engineering

Instead of feeding raw board pixels, we extracted 75 distinct features per board state:

Board State: One-hot encoded board representation (64 squares).

Material Imbalance: Heuristic sum (Q=9, R=5, etc.).

Mobility Imbalance: The differential in legal moves available to White vs. Black. This was identified as the 2nd most important feature in Random Forest analysis.

Data Splitting

We used Grouped Splitting by Game ID.

Problem: Randomly splitting rows causes data leakage (Move 15 in Train, Move 16 in Test).

Solution: We ensured that if Game X is in the training set, all moves from Game X are in the training set.

📈 Results Visualization

The project generates several key plots in the plots/ directory:

rf_feature_importance.png: Visual proof of Mobility's impact.

logreg_coefficients.png: Directional impact of features (Positive = White Advantage).

dl_training_history.png: Learning curves showing the MLP's convergence.

