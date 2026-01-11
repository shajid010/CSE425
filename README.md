# Music Clustering VAE

Unsupervised learning pipeline for clustering music using Variational Autoencoders (VAE).

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic VAE (Vector Input)
Train a simple VAE on synthetic vector data (representing features like MFCC means).
```bash
python -m src.main --mode train --model_type linear --epochs 50 --n_clusters 5
```

### 2. Baseline Comparison
Run PCA + K-Means for comparison.
```bash
python -m src.baseline --n_components 10 --n_clusters 5
```

### 3. Convolutional VAE (Spectrograms) 
```bash
python -m src.main --mode train --model_type conv --epochs 50
```

### 4. Hybrid VAE (Audio + Lyrics) 
```bash
python -m src.main --mode train --model_type hybrid --epochs 50
```

## Results
Results (plots, metrics) are saved in the `results/` directory.
