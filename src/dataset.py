import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MusicDataset(Dataset):
    """
    Dataset class for Music VAE.
    Supports both synthetic data and loading from pre-processed files.
    """
    def __init__(self, data, labels=None):
        if isinstance(data, tuple):
            # Hybrid mode: (audio, lyrics)
            self.data = data # Tuple of arrays
            self.is_tuple = True
            self.length = len(data[0])
        else:
            self.data = torch.FloatTensor(data)
            self.is_tuple = False
            self.length = len(data)
            
        self.labels = labels if labels is not None else None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_tuple:
            audio = torch.FloatTensor(self.data[0][idx])
            lyrics = torch.FloatTensor(self.data[1][idx])
            sample = (audio, lyrics)
        else:
            sample = self.data[idx]
            
        if self.labels is not None:
            return sample, self.labels[idx]
        return sample

class SyntheticDataGenerator:
    """
    Generates synthetic music feature data for testing.
    Simulates clusters to verify VAE disentanglement/clustering capabilities.
    """
    def __init__(self, n_samples=1000, n_features=128, n_classes=5, mode='linear'):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.mode = mode

    def generate(self):
        if self.mode == 'linear':
            return self._generate_linear()
        elif self.mode == 'conv':
            return self._generate_conv()
        elif self.mode == 'hybrid':
            return self._generate_hybrid()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _generate_linear(self):
        # Generate centers for clusters
        centers = np.random.uniform(-5, 5, size=(self.n_classes, self.n_features))
        X = []
        y = []
        samples_per_class = self.n_samples // self.n_classes
        for i in range(self.n_classes):
            noise = np.random.normal(0, 1.0, size=(samples_per_class, self.n_features))
            class_samples = centers[i] + noise
            X.append(class_samples)
            y.extend([i] * samples_per_class)
        X = np.vstack(X)
        y = np.array(y)
        return self._shuffle(X, y)

    def _generate_conv(self):
        # Generate 64x64 "spectrograms"
        # We'll just create patterns: horizontal bars, vertical bars, random blobs
        X = np.zeros((self.n_samples, 1, 64, 64))
        y = np.zeros(self.n_samples, dtype=int)
        
        samples_per_class = self.n_samples // self.n_classes
        for i in range(self.n_samples):
            label = i // samples_per_class
            y[i] = label
            
            # Simple synthetic patterns based on class
            if label % 3 == 0: # Horizontal lines
                row = np.random.randint(0, 64)
                X[i, 0, row:row+5, :] = 1.0
            elif label % 3 == 1: # Vertical lines
                col = np.random.randint(0, 64)
                X[i, 0, :, col:col+5] = 1.0
            else: # Blobs
                r, c = np.random.randint(0, 54, 2)
                X[i, 0, r:r+10, c:c+10] = 1.0
                
            # Add noise
            X[i] += np.random.normal(0, 0.1, (1, 64, 64))
            
        return self._shuffle(X, y)

    def _generate_hybrid(self):
        # Audio features + "Lyrics" features (Bag of Words style vectors)
        audio_dim = self.n_features
        lyrics_dim = 100
        
        # Re-use linear generation for both, just concatenated conceptual logic
        X_audio, y = self._generate_linear()
        
        # Generate correlated lyrics features
        # For simplicity, let's just make lyrics features also cluster-dependent
        centers_lyrics = np.random.uniform(-3, 3, size=(self.n_classes, lyrics_dim))
        X_lyrics = []
        
        # We need to ensure we match the labels from X_audio if we want them correlated. 
        # But _generate_linear returns X,y combined/shuffled.
        # So we should rewrite a bit to handle indices or generation order.
        
        # Let's do a cleaner build:
        X_audio_list = []
        X_lyrics_list = []
        y_list = []
        samples_per_class = self.n_samples // self.n_classes
        
        centers_audio = np.random.uniform(-5, 5, size=(self.n_classes, audio_dim))
        
        for i in range(self.n_classes):
            # Audio
            noise_a = np.random.normal(0, 1.0, size=(samples_per_class, audio_dim))
            X_audio_list.append(centers_audio[i] + noise_a)
            
            # Lyrics
            noise_l = np.random.normal(0, 1.0, size=(samples_per_class, lyrics_dim))
            X_lyrics_list.append(centers_lyrics[i] + noise_l)
            
            y_list.extend([i] * samples_per_class)
            
        X_audio = np.vstack(X_audio_list)
        X_lyrics = np.vstack(X_lyrics_list)
        y = np.array(y_list)
        
        # Shuffle together
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        
        X_audio = X_audio[indices]
        X_lyrics = X_lyrics[indices]
        y = y[indices]
        
        return (X_audio, X_lyrics), y

    def _shuffle(self, X, y):
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        return X[indices], y[indices]

def get_dataloader(batch_size=32, synthetic=True, mode='linear'):
    if synthetic:
        generator = SyntheticDataGenerator(mode=mode)
        X, y = generator.generate()
        
        # X might be a tuple for hybrid
        dataset = MusicDataset(X, y)
    else:
        raise NotImplementedError("Real data loading not implemented yet")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), X, y
