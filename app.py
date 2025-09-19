import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -------------------------------
# RBM Model
# -------------------------------
class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=64):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob, torch.bernoulli(prob)

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            prob_h, h = self.sample_h(v)
            prob_v, v = self.sample_v(h)
        return v0, v, prob_h

# -------------------------------
# Training function
# -------------------------------
@st.cache_resource
def train_rbm(epochs=1, batch_size=64, n_hidden=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    rbm = RBM(n_visible=784, n_hidden=n_hidden)
    optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)

    for epoch in range(epochs):
        for batch, _ in trainloader:
            v0, vk, ph = rbm.contrastive_divergence(batch, k=1)
            loss = torch.mean((v0 - vk) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return rbm

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 Restricted Boltzmann Machine (RBM) Visualizer")

epochs = st.sidebar.slider("Epochs", 1, 5, 1)
hidden_units = st.sidebar.slider("Hidden Units", 16, 128, 64)

st.write(f"Training RBM with {hidden_units} hidden units for {epochs} epoch(s)...")
rbm = train_rbm(epochs=epochs, n_hidden=hidden_units)

# -------------------------------
# Show hidden weights
# -------------------------------
st.header("🔍 Hidden Unit Weight Filters")
W = rbm.W.detach().cpu()

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    if i < hidden_units:
        ax.imshow(W[i].view(28, 28), cmap="gray")
        ax.axis("off")
st.pyplot(fig)

# -------------------------------
# Hidden activation heatmap
# -------------------------------
st.header("🔥 Hidden Layer Activation Heatmap")
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

images, _ = next(iter(testloader))
v = images.view(-1)
prob_h, _ = rbm.sample_h(v)

plt.figure(figsize=(10, 2))
plt.imshow(prob_h.detach().numpy().reshape(1, -1), cmap="hot", aspect="auto")
plt.colorbar()
plt.title("Hidden Unit Activations")
st.pyplot(plt)
