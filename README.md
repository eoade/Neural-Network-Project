# From Tables to Images — A Neural Network Lab

## Project overview

This project explores how neural networks learn from two kinds of data:

- **Structured data:** Framingham patient records, using the `TenYearCHD` target.
- **Unstructured data:** Fashion-MNIST grayscale images across ten clothing categories.

The project investigates model performance, underfitting, overfitting, generalization, image classification, and representation learning. It follows the Week 6 project brief from the 3MTT x MIT Universal AI programme.

> **Important:** This is an educational machine-learning exercise. The Framingham model is not a clinically validated diagnostic tool and must not be used to make medical decisions.

## Project stages and results

### Stage 1 — Neural networks for structured data

The cleaned Framingham dataset was reviewed for missing values, target-class balance, and available numeric features. A Perceptron, a one-hidden-layer multilayer perceptron (MLP), and logistic regression were trained and compared.

The dataset contained **4,240 records** and **15 numeric input features**. The target was imbalanced: **84.81%** of records had `TenYearCHD = 0`, while **15.19%** had `TenYearCHD = 1`.

Missing values were most common in glucose (388), education (105), BPMeds (53), total cholesterol (50), and cigarettes per day (29). Numeric missing values were handled using median imputation. Imputation and standardization were fitted on the training data only.

| Model | Training error | Test error | Test AUC |
|---|---:|---:|---:|
| Perceptron | 26.81% | 25.94% | — |
| MLP (one hidden layer) | 14.54% | 15.68% | 0.6860 |
| Logistic regression | — | 15.33% | 0.7043 |

Logistic regression slightly outperformed the tested MLP by test AUC, showing that a more complex model is not automatically better.

### Stage 2 — Underfitting, overfitting, and generalization

MLP models were tested with hidden-layer sizes of 1, 4, 16, 64, and 256 neurons. Training and validation errors were compared to examine the effect of model complexity.

- Training error generally decreased from **15.17%** with 1 neuron to **13.76%** with 256 neurons.
- The lowest validation error among the tested sizes was **14.39%**, recorded with 256 neurons.
- The tested regularization values (`alpha` = 0.0001, 0.001, 0.01, and 0.1) produced the same recorded training error (**14.23%**) and validation error (**15.21%**) in this run.
- Adding pulse pressure (`sysBP - diaBP`) left validation accuracy at **84.79%** and changed test accuracy from **84.32%** to **84.79%**.

The results emphasize the importance of validation performance when selecting a model. In this run, the tested regularization values did not produce a measurable change in the recorded errors.

### Stage 3 — Image classification with Fashion-MNIST

Fashion-MNIST was used to compare a dense neural network with a convolutional neural network (CNN). The MLP received flattened images with 784 pixel inputs. The CNN used convolution and pooling layers to learn local image patterns. A confusion matrix and examples of misclassified images were also examined.

| Model | Test accuracy | Parameters | Training time |
|---|---:|---:|---:|
| MLP (flattened pixels) | 82.88% | 101,770 | 3.7 seconds |
| CNN | 84.45% | 220,234 | 72.4 seconds |

The CNN achieved higher test accuracy, but used more parameters and took longer to train. Errors were common among visually similar upper-body categories, including shirts, T-shirts/tops, pullovers, and coats.

To keep training manageable, the notebook used a **10,000-image training subset** and trained the models for five epochs.

### Stage 4 — Representation learning and embeddings

An autoencoder was trained to compress and reconstruct Fashion-MNIST images using a two-dimensional latent representation. Original and reconstructed images were displayed, reconstruction error was measured, and the latent space was visualized. K-means clustering was then applied to the latent representations.

The recorded reconstruction mean squared error on 20 test images was **0.040547**. The latent representation showed some grouping, but the k-means clusters contained substantial mixing between true classes. This illustrates that a representation trained to reconstruct images is not necessarily optimized to separate semantic categories.

The notebook also compares CNNs, autoencoders, and GANs, and explains how text embeddings differ from one-hot encoding.

## Key findings

1. Logistic regression slightly outperformed the tested MLP on Framingham test AUC (0.7043 versus 0.6860).
2. The Perceptron had higher training and test error than the MLP and logistic regression.
3. Increasing hidden-layer size reduced training error; validation error was used to compare generalization.
4. The tested regularization values produced identical recorded errors in this run.
5. Adding pulse pressure produced a small increase in test accuracy while validation accuracy remained unchanged.
6. The CNN achieved higher Fashion-MNIST test accuracy than the flattened-pixel MLP.
7. The CNN required more parameters and substantially more training time.
8. Similar clothing shapes were more difficult for the image classifier to distinguish.
9. The autoencoder achieved a reconstruction MSE of 0.040547, but its latent clusters did not cleanly separate all ten classes.
10. Overall, the experiments show that greater model complexity does not guarantee better performance on every dataset. Validation-based model selection and careful interpretation are essential.

## Repository contents

```text
.
├── Neural_Networks_Project_Completed.ipynb
└── README.md
```

The notebook contains the code, analysis, tables, and visualizations for all four stages, together with the completed findings summary and stage reflections.

## How to run the notebook

### Recommended: Google Colab

1. Download `Neural_Networks_Project_Completed.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload the notebook.
4. Run the cells in order.
5. When prompted, upload the cleaned Framingham CSV dataset from Week 5.
6. Allow Keras to download Fashion-MNIST when the image-classification stage runs.

### Local Jupyter environment

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

Then open the notebook:

```bash
jupyter notebook Neural_Networks_Project_Completed.ipynb
```

The Framingham CSV should be available in the notebook's working directory or selected when prompted. The target column should be named `TenYearCHD` or use a recognized equivalent.

## Reproducibility and limitations

- Random seeds are set in the notebook to support repeatability.
- Preprocessing is fitted on training data only.
- Validation data is used for model comparison and selection; the test set is reserved for final evaluation within each stage.
- Small differences may occur across hardware and library versions.
- The results above describe the saved notebook run. Rerunning the notebook may produce different training times or slightly different metrics.
- The Framingham target is imbalanced, and dataset sampling and label limitations should be considered when interpreting performance.
- Fashion-MNIST training used a subset to reduce runtime.

## Acknowledgement

This project follows the Week 6 project brief, **“From Tables to Images, a Neural Network Lab,”** from the 3MTT x MIT Universal AI programme.
