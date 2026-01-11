import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, latent_dim=10):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        return self.encoder_fc2_mean(h), self.encoder_fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        return self.decoder_fc2(h) # No activation for raw features, or Sigmoid if normalized [0,1]
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class ConvVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=10):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flatten size: 128 * 8 * 8 = 8192 (assuming 64x64 input)
        self.fc_mean = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # 32x32 -> 64x64
            # No Sigmoid if data is raw features, but often Spectrograms are [0,1] or log-magnitude. 
            # We'll assume normalized data for now.
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class HybridVAE(nn.Module):
    def __init__(self, audio_dim=128, lyrics_dim=100, hidden_dim=64, latent_dim=10):
        super(HybridVAE, self).__init__()
        
        # Audio Encoder
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Lyrics Encoder
        self.lyrics_enc = nn.Sequential(
            nn.Linear(lyrics_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoders
        self.decoder_input = nn.Linear(latent_dim, hidden_dim * 2)
        
        self.audio_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim)
        )
        
        self.lyrics_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lyrics_dim)
        )
        
    def encode(self, audio, lyrics):
        h_audio = self.audio_enc(audio)
        h_lyrics = self.lyrics_enc(lyrics)
        h = torch.cat([h_audio, h_lyrics], dim=1)
        return self.fc_mean(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.decoder_input(z)
        # Split back to audio/lyrics paths
        h_audio, h_lyrics = torch.chunk(h, 2, dim=1)
        return self.audio_dec(h_audio), self.lyrics_dec(h_lyrics)
        
    def forward(self, x):
        # x is a list/tuple: (audio, lyrics)
        audio, lyrics = x
        mu, logvar = self.encode(audio, lyrics)
        z = self.reparameterize(mu, logvar)
        recon_audio, recon_lyrics = self.decode(z)
        return (recon_audio, recon_lyrics), mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_dim=128, n_classes=5, hidden_dim=64, latent_dim=10):
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # Encoder
        # Input: x + label (one-hot)
        self.encoder_fc1 = nn.Linear(input_dim + n_classes, hidden_dim)
        self.encoder_fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        # Input: z + label (one-hot)
        self.decoder_fc1 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x, y):
        # y is class index, need to one-hot encode
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float().to(x.device)
        inputs = torch.cat([x, y_onehot], dim=1)
        h = F.relu(self.encoder_fc1(inputs))
        return self.encoder_fc2_mean(h), self.encoder_fc2_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, y):
        y_onehot = F.one_hot(y, num_classes=self.n_classes).float().to(z.device)
        inputs = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.decoder_fc1(inputs))
        return self.decoder_fc2(h)
    
    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction term
    if isinstance(recon_x, tuple) and isinstance(x, tuple):
        # Hybrid case
        MSE = 0
        for rx, tx in zip(recon_x, x):
            MSE += F.mse_loss(rx, tx, reduction='sum')
    else:
        # Standard case
        MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence term
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta-VAE loss
    return MSE + beta * KLD
