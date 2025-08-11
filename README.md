# Probabilistic Deep Learning

A collection of Jupyter notebooks illustrating core ideas and techniques in **probabilistic machine learning** using **TensorFlow** and **TensorFlow Probability**.  
Each notebook is a hands-on implementation of a specific model or inference method — from simple Naive Bayes to Bayesian Neural Networks, Variational Autoencoders, and Normalizing Flows.

---

## 📚 Contents

### 1. Naive Bayes
- Implements a basic probabilistic classifier using Bayes’ theorem.
- Covers likelihood estimation and class posterior computation.

### 2. Bayesian Neural Networks (BNNs)
- Neural networks with **probabilistic weights**.
- Uses TensorFlow Probability’s `tfp.layers` and `tfp.layers.DistributionLambda`.
- Demonstrates uncertainty estimation (aleatoric and epistemic).

### 3. Variational Autoencoders (VAEs)
- Generative latent-variable models trained via the **ELBO** (Evidence Lower Bound).
- Implements the **reparameterization trick** for backpropagation through random variables.
- Generates samples from the learned latent space.

### 4. Normalizing Flows
- Density estimation models using a **change of variables** formula.
- Trains flexible distributions via invertible transformations.
- Explores coupling layers and Jacobian log-determinants.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/lqrz/probabilistic_deep_learning.git
cd probabilistic_deep_learning


---

## Dependencies
```bash
pip install tensorflow tensorflow-probability numpy matplotlib jupyter

