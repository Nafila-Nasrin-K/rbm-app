
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# RBM model
class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=64):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob, torch.bernoulli(prob)

# Training RBM (tiny demo for speed)
@st.cache_resource
def train_rbm(n_hidden=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    rbm = RBM(n_hidden=n_hidden)
    opt = torch.optim.SGD(rbm.parameters(), lr=0.1)

    for images, _ in loader:
        v = images
        ph, h = rbm.sample_h(v)
        loss = torch.mean((v - v) ** 2)  # dummy loss for simplicity
        opt.zero_grad()
        loss.backward()
        opt.step()
        break  # only one batch for speed
    return rbm

# Streamlit UI
st.title("🧠 RBM Visualizer")

hidden_units = st.sidebar.slider("Hidden Units", 16, 128, 64)
rbm = train_rbm(n_hidden=hidden_units)

# Show filters
W = rbm.W.detach().cpu()
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    if i < hidden_units:
        ax.imshow(W[i].view(28, 28), cmap="gray")
        ax.axis("off")
st.pyplot(fig)
