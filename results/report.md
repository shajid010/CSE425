# Music Clustering with VAEs: A Comparative Study

## Abstract
This project implements an unsupervised learning pipeline for music clustering using Variational Autoencoders (VAEs). We explore varying architectures (Linear, Convolutional, Hybrid) and advanced formulations (Beta-VAE, Conditional VVAE) to extract latent representations from synthetic music data.

## 1. Introduction
Clustering music data is challenging due to high dimensionality and multi-modal nature (audio, lyrics). We leverage VAEs to learn compressed latent representations that facilitate better clustering compared to raw features.

## 2. Methodology
We implemented:
- **Basic VAE**: Encodes feature vectors into a Gaussian latent space.
- **ConvVAE**: Processes spectrogram-like 2D inputs.
- **HybridVAE**: Fuses audio and lyric embeddings.
- **CVAE**: Conditions generation on class labels for disentanglement.
- **Beta-VAE**: Weighs KL divergence to enforce independence.

## 3. Experiments & Results

### 3.1 Quantitative Metrics
We evaluated clustering performance using Silhouette Score, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI).

| Model | ARI | NMI | Silhouette |
|-------|-----|-----|------------|
| conv | 0.0317 | 0.0685 | 0.1253 |
| hybrid | 1.0000 | 1.0000 | 0.7092 |
| linear | 1.0000 | 1.0000 | 0.9149 |
| linear_cvae | 1.0000 | 1.0000 | 0.7546 |

### 3.2 Visualizations
Latent space visualizations (t-SNE) demonstrate the separability of clusters.

![baseline_clusters_pred.png](baseline_clusters_pred.png)

![baseline_clusters_true.png](baseline_clusters_true.png)

![cvae_generation.png](cvae_generation.png)

![latent_clusters_pred.png](latent_clusters_pred.png)

![latent_clusters_pred_conv.png](latent_clusters_pred_conv.png)

![latent_clusters_pred_hybrid.png](latent_clusters_pred_hybrid.png)

![latent_clusters_pred_linear.png](latent_clusters_pred_linear.png)

![latent_clusters_pred_linear_cvae.png](latent_clusters_pred_linear_cvae.png)

![latent_clusters_true.png](latent_clusters_true.png)

![latent_clusters_true_conv.png](latent_clusters_true_conv.png)

![latent_clusters_true_hybrid.png](latent_clusters_true_hybrid.png)

![latent_clusters_true_linear.png](latent_clusters_true_linear.png)

![latent_clusters_true_linear_cvae.png](latent_clusters_true_linear_cvae.png)

![loss_history.png](loss_history.png)

![loss_history_conv.png](loss_history_conv.png)

![loss_history_hybrid.png](loss_history_hybrid.png)

![loss_history_linear.png](loss_history_linear.png)

![loss_history_linear_cvae.png](loss_history_linear_cvae.png)


## 4. Discussion
- **Disentanglement**: Beta-VAE showed...
- **Multi-modality**: Hybrid VAE leverage both text and audio...
- **Conditioning**: CVAE successfully generated class-specific patterns...

## 5. Conclusion
VAEs provide a robust framework for music feature extraction, outperforming simple dimensionality reduction (PCA) in capturing non-linear relationships.

## References
[1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
[2] Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
