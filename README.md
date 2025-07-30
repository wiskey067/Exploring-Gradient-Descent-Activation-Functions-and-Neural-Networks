Neural Network Optimization & Activation Function Analysis


Author: Arijit Bhattacharjee


🧠 Overview

This project explores core components of neural networks through hands-on experimentation and analysis using the sklearn digits dataset. Key focus areas include:
	•	Custom implementation of gradient descent
	•	Comparison of sigmoid, tanh, and ReLU activation functions
	•	Numerical vs. analytical gradient checking
	•	Neural network training using different optimizers (SGD, Adam, LBFGS)
	•	Performance comparison in terms of accuracy, loss convergence, and training time

⸻

📁 Table of Contents
	1.	Introduction
	2.	Experimental Setup
	3.	Visualizations
	4.	Results Summary
	5.	Key Findings
	6.	Conclusion
	7.	Appendix

⸻

🔬 Introduction

This project investigates how optimization techniques and activation functions affect neural network performance for a binary classification task. We classify handwritten digits from the sklearn.datasets.load_digits() dataset, relabeled as:
	•	Class 0: Digits 0–4
	•	Class 1: Digits 5–9

The project includes:
	•	Custom implementation of gradient descent
	•	Analytical and numerical gradient verification
	•	Model training using MLPClassifier with various solvers
	•	Visual comparison of activation functions and optimizers

⸻

⚙️ Experimental Setup
	•	Dataset: sklearn.datasets.load_digits()
	•	Binary Labels: 0–4 → 0, and 5–9 → 1
	•	Preprocessing: Standardization and 80-20 Train/Test Split
	•	Libraries Used: NumPy, Matplotlib, scikit-learn
	•	Network Type: Feedforward MLP
	•	Solvers Tested: SGD, Adam, LBFGS
	•	Custom Modules:
	•	Gradient Descent
	•	Activation functions and derivatives
	•	Gradient checking (numerical vs. backprop)

⸻

📊 Visualizations

🔁 Activation Functions & Their Gradients
	•	Functions: Sigmoid, Tanh, ReLU
	•	Observation:
	•	ReLU maintains strong gradients
	•	Sigmoid and Tanh suffer from vanishing gradients in deeper layers

📉 Loss Curves (Adam vs. SGD)
	•	Adam converges faster and more consistently
	•	SGD shows slower convergence with more fluctuations

🎯 Accuracy and Time Comparison
	•	Adam and LBFGS achieve high accuracy in less time
	•	SGD is computationally slower but stable

⸻

📋 Results Summary

Optimizer	Train Accuracy	Test Accuracy	Training Time
SGD	~97%	~94%	High
Adam	~99%	~96%	Medium
LBFGS	~99%	~95%	Lowest

(Exact values may vary slightly with random seed)

⸻

📌 Key Findings

Optimizer Performance
	•	SGD: Stable but slow; good for small-scale training
	•	Adam: Best balance of speed and accuracy
	•	LBFGS: Fastest but not ideal for large datasets

Activation Functions
	•	ReLU performed best, especially in deeper networks
	•	Sigmoid/Tanh were slower and susceptible to vanishing gradients

General Observations
	•	Overfitting Risk: Slight overfitting observed in LBFGS
	•	Gradient Checking: Analytical gradients matched numerical ones
	•	Performance Bottlenecks: Manual gradient computation is slow; use auto-diff frameworks for scaling

⸻

✅ Conclusion
	•	Best Combo: Adam + ReLU for this classification task
	•	For Larger Networks: Monitor for overfitting, especially with fast optimizers
	•	Custom Code Insights: Great for educational use; inefficient for large-scale production training

⸻

📎 Appendix
	•	Notebook: gradient_descent_neural_net.ipynb
	•	Libraries Used:
	•	numpy
	•	matplotlib
	•	sklearn
	•	All plots, tables, and logs included in the notebook



📌 How to Run

# Clone the repo
git clone https://github.com/wiskey067/gradient-descent-nn-experiments.git
cd gradient-descent-nn-experiments

# Launch Jupyter
jupyter notebook gradient_descent_neural_net.ipynb




🙌 Acknowledgements

Thanks to the scikit-learn and numpy communities for open-source contributions that made this analysis possible.
