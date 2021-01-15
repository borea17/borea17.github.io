---
title: "Importance Weighted Autoencoders"
permalink: "/paper_summaries/iwae"
author: "Markus Borea"
tags: ["generative model"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

[Burda et al. (2016)](https://arxiv.org/abs/1509.00519) introduce the importance
weighted autoencoder (IWAE) as a simple modification in the training
of [variational autoencoders](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes)
which derives from importance weighting and leads to a strictly tighter evidence
lower bound (ELBO).

-



## Model Description

### Motivation

### Derivation




## Implementation

{% capture code %}{% raw %}class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_SIZE**2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2*LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, WINDOW_SIZE**2)
        )
        return

    def compute_loss(self, x):
        [x_tilde, z, mu_z, log_var_z] = self.forward(x)
        NLL = (1/(2*VAR_FIXED))*(x - x_tilde).pow(2).sum((1,2,3)).mean()
        KL_Div = -0.5*(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(1).mean()
        return NLL, KL_Div

    def forward(self, x):
        z, mu_z, log_var_z = self.encode(x)
        x_tilde = self.decode(z)
        return [x_tilde, z, mu_z, log_var_z]

    def encode(self, x):
        # get encoder distribution parameters
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # sample noise variable for each batch
        epsilon = torch.randn_like(log_var_E)
        # get latent variable by reparametrization trick
        z = mu_E + torch.exp(0.5*log_var_E) * epsilon
        return z, mu_E, log_var_E

    def decode(self, z):
        # get decoder distribution parameters
        x_tilde = self.decoder(z)
        # reshape into [batch, 1, WINDOW_SIZE, WINDOW_SIZE] (input shape)
        x_tilde = x_tilde.view(-1, 1, WINDOW_SIZE, WINDOW_SIZE)
        return x_tilde{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}


{% capture code %}{% raw %}class IWAE(nn.Module):

    def __init__(self, k):
        super(IWAE, self).__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_SIZE**2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2*LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, WINDOW_SIZE**2)
        )
        return

    def compute_loss(self, x):
        [x_tilde, z, mu_z, log_var_z] = self.forward(x)
        # make x size as x_tilde [batch, samples, 1, WINDOW_SIZE, WINDOW_SIZE]
        x = x.unsqueeze(1).repeat(1, self.k, 1, 1, 1)
        # compute NLL [batch, samples]
        NLL = (1/(2*VAR_FIXED))*(x - x_tilde).pow(2).sum(axis=(2, 3, 4))
        # compute KL div [batch, samples]
        KL_Div =  -0.5*(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(2)
        # get importance weights [in log units] [batch, samples]
        log_weights = (NLL + KL_Div)
        # get tilde{w} (sum_k \log {\tilde{w}} = 1)
        w_tilde = F.softmax(log_weights, dim=1)
        # compute loss
        loss = (w_tilde*log_weights).sum(1).mean()

        NLL_VAE = NLL[:, 0].mean()
        KL_Div_VAE =  KL_Div[:,0].mean()
        return loss, NLL_VAE, KL_Div_VAE

    def forward(self, x):
        z, mu_z, log_var_z = self.encode(x)
        x_tilde = self.decode(z)
        return [x_tilde, z, mu_z, log_var_z]

    def encode(self, x):
        # get encoder distribution parameters
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # increase shape for sampling [batch, samples, latent_dim]
        mu_E = mu_E.view(x.shape[0], 1, -1).repeat(1, self.k, 1)
        log_var_E = log_var_E.view(x.shape[0], 1, -1).repeat(1, self.k, 1)
        # sample noise variable for each batch and sample
        epsilon = torch.randn_like(log_var_E)
        # get latent variable by reparametrization trick
        z = mu_E + torch.exp(0.5*log_var_E) * epsilon
        return z, mu_E, log_var_E

    def decode(self, z):
        batch_size = z.shape[0]
        # parallelize computation by stacking samples along batch dim
        z = z.view(-1, z.shape[2])  # [batch*samples, latent_dim]
        # get decoder distribution parameters
        x_tilde = self.decoder(z)  # [batch*samples, WINDOW_SIZE**2]
        # reshape into [batch, samples, 1, WINDOW_SIZE, WINDOW_SIZE] (input shape)
        x_tilde = x_tilde.view(batch_size, self.k, 1, WINDOW_SIZE, WINDOW_SIZE)
        return x_tilde{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}
