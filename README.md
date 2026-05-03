# Deep Learning Notebooks

A collection of 5 hands-on Jupyter notebooks that walk through the core deep learning architectures: feedforward neural networks, convolutional neural networks (CNN), recurrent neural networks (RNN), and long short-term memory networks (LSTM). Each lab uses a real dataset and follows the full workflow from data loading to evaluation.

---

## Contents

| # | Notebook | Architecture | Task | Dataset |
|---|---|---|---|---|
| 01 | `01_lab.ipynb` | Dense Neural Network | Binary classification | Diabetes |
| 02 | `02_lab.ipynb` | Convolutional Neural Network | Image classification | CIFAR-10 |
| 03 | `03_lab.ipynb` | Simple RNN | Sentiment analysis | IMDB reviews |
| 04 | `04_lab.ipynb` | Simple RNN | Time series forecasting | Beijing PM2.5 |
| 05 | `05_lab.ipynb` | LSTM | Stock price prediction | Google (GOOG) |

---

## Requirements

```
pandas >= 2.0.0
numpy >= 1.24.0
tensorflow >= 2.10.0
scikit-learn >= 1.2.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
tqdm >= 4.64.0
pydantic >= 1.10.0
PyYAML >= 6.0
```

Install with:

```bash
pip install -r req.txt
```

> 💡 **GPU recommended.** Lab 02 (CNN on CIFAR-10) and Lab 05 (LSTM) train much faster on a GPU. They will work on CPU but expect long training times.

---

## Quick Start

```bash
# Clone or download the project
cd deep-learning-notebooks

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r req.txt

# Launch Jupyter
jupyter notebook
```

Open any notebook and run the cells in order.

---

## Notebook Summaries

### 01 — Dense Neural Network (Diabetes Prediction)

A starter project: predict whether a patient has diabetes from clinical features. Compare a Random Forest baseline against simple neural networks of increasing depth.

**What you'll learn:**
- Building a model with `Sequential` and `Dense` layers
- The role of activation functions (sigmoid, ReLU)
- Training with `model.fit()` and evaluating with `model.evaluate()`
- Comparing a deep model against a classical baseline (Random Forest)
- Plotting the **ROC curve** and computing **AUC**
- Why feature scaling matters for neural networks (`MinMaxScaler`)

---

### 02 — CNN (CIFAR-10 Image Classification)

Classify 32×32 color images into 10 categories (airplane, car, bird, cat, dog, ...). The lab walks through a modern CNN with batch normalization, dropout, and data augmentation.

**What you'll learn:**
- The convolutional building blocks: `Conv2D`, `MaxPooling2D`
- **Batch Normalization** — stabilizes training and speeds up convergence
- **Dropout** — random neuron deactivation to fight overfitting
- **Data Augmentation** with `ImageDataGenerator` (random shifts, flips)
- One-hot encoding labels with `to_categorical`
- Smart training callbacks:
  - `ReduceLROnPlateau` — automatic learning rate reduction
  - `EarlyStopping` — stop when validation stops improving
- Visualizing predictions with probability bars

---

### 03 — Simple RNN (IMDB Sentiment Analysis)

Classify movie reviews as positive or negative using a recurrent network that reads text word by word.

**What you'll learn:**
- How RNNs process **sequential data**
- **Embedding layers** — turn words into dense vectors
- **Padding sequences** to a uniform length with `pad_sequences`
- The IMDB dataset's special token offsets (`<PAD>`, `<START>`, `<UNK>`)
- Building a custom inference function that takes raw text and outputs a sentiment label
- Why `SimpleRNN` struggles with long sequences (vanishing gradients)

---

### 04 — RNN for Time Series (Beijing Air Quality)

Forecast PM2.5 air pollution levels in Beijing from past hourly readings.

**What you'll learn:**
- Loading and indexing time series data
- **Linear interpolation** to fill missing values
- Constructing **sliding windows** for sequence-to-one prediction
- Reshaping data to RNN's required shape: `(samples, timesteps, features)`
- **Recursive forecasting** — using a model's own prediction as the next input
- Why error compounds in multi-step forecasts

---

### 05 — LSTM (Google Stock Price Forecasting)

Predict the next-day closing price of Google stock using an LSTM network with engineered features (moving averages, returns, volatility).

**What you'll learn:**
- The LSTM architecture and its three gates (forget, input, output)
- **Feature engineering for finance:**
  - Moving averages (`ma_5`, `ma_20`)
  - Daily returns (`pct_change`)
  - Rolling volatility
- Stacking LSTM layers with `return_sequences=True/False`
- Predicting **returns** instead of raw prices, then converting back
- Chronological train/test splitting (no shuffle for time series)
- Reproducibility with `np.random.seed` and `tf.keras.utils.set_random_seed`

---

## Common Workflow

Every notebook follows roughly this pattern:

```
1. Import libraries
2. Load and explore data
3. Preprocess (scale, encode, reshape, split)
4. Build the model (Sequential + layers)
5. Compile (optimizer, loss, metrics)
6. Train (model.fit)
7. Evaluate (metrics, plots)
8. Predict on new data
```

Once you internalize this pattern, you can apply it to any new dataset.

---

## Key Concepts Reference

### Activation Functions

| Where | Activation | Why |
|---|---|---|
| Hidden layers (default) | **ReLU** | Fast, avoids vanishing gradients |
| Output: binary classification | **sigmoid** | Squashes to 0–1 (probability) |
| Output: multi-class | **softmax** | Probabilities that sum to 1 |
| Output: regression | **None (linear)** | Unbounded numerical output |

### Loss Functions

| Task | Loss |
|---|---|
| Binary classification | `binary_crossentropy` |
| Multi-class (one-hot labels) | `categorical_crossentropy` |
| Multi-class (integer labels) | `sparse_categorical_crossentropy` |
| Regression | `mse` or `mae` |

### Common Optimizers

- **SGD** — classic, slow but generalizes well
- **Adam** — fast and reliable, the default choice for most tasks
- **RMSprop** — historically popular for RNNs

### Useful Callbacks

- **EarlyStopping** — halt training when validation loss stops improving
- **ReduceLROnPlateau** — lower the learning rate when stuck
- **ModelCheckpoint** — save the best model during training

---

## Tips and Best Practices

- **Always scale your inputs** when using neural networks. Use `MinMaxScaler` for [0, 1] range or `StandardScaler` for zero-mean, unit-variance data.
- **Set random seeds** for reproducibility:
  ```python
  np.random.seed(42)
  tf.keras.utils.set_random_seed(42)
  ```
- **Don't shuffle time series data.** Past must come before future. Use `shuffle=False` in `model.fit()` for time series.
- **Don't peek at test data during training.** Use `validation_split` carved from training data — not the test set.
- **Watch for overfitting.** If training accuracy ↑ but validation accuracy ↓, add dropout, augmentation, or stop earlier.
- **Compare against a baseline.** A neural network is only useful if it beats a simple model (Random Forest, naive forecast, majority class).

---

## Project Structure

```
deep-learning-notebooks/
├── 01_lab.ipynb              # Dense NN
├── 02_lab.ipynb              # CNN
├── 03_lab.ipynb              # RNN (text)
├── 04_lab.ipynb              # RNN (time series)
├── 05_lab.ipynb              # LSTM
├── data/
│   ├── diabetes.csv
│   ├── Beijing.csv
│   └── GOOG.csv
├── req.txt
└── README.md
```

> Place your dataset files in a `data/` folder and update file paths in each notebook to use relative paths like `data/diabetes.csv` for portability. CIFAR-10 is downloaded automatically by Keras.

---

## Notes

- Lab paths in some notebooks are Windows-specific. Adjust them or use relative paths.
- TensorFlow may print deprecation warnings — they're safe to ignore for these labs.
- The CIFAR-10 dataset is ~170 MB and downloads automatically the first time you run Lab 02.

---

## License

This project is provided for educational purposes.

---

## Acknowledgments

Built with `tensorflow` (Keras), `scikit-learn`, `pandas`, `numpy`, and `matplotlib`. Datasets are public and used for learning purposes only.
