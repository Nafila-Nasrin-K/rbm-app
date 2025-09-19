import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="RBM Playground Animation", layout="centered")

st.title("🔮 RBM Playground (Animated Steps)")
st.write("Step through how an RBM works: **Visible → Hidden → Reconstruction**")

# --- User Settings ---
n_visible = st.slider("Number of Visible Units", 3, 8, 6)
n_hidden = st.slider("Number of Hidden Units", 2, 6, 3)

# Initialize weights
np.random.seed(42)
W = np.random.randn(n_visible, n_hidden) * 0.5

# --- Input Selection ---
st.subheader("Step 1: Choose Input (Visible Units)")
visible = []
cols = st.columns(n_visible)
for i in range(n_visible):
    visible.append(cols[i].checkbox(f"v{i+1}", value=(i % 2 == 0)))
v = np.array(visible, dtype=int)

# --- Sigmoid ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

hidden_probs = sigmoid(np.dot(v, W))
hidden_states = (hidden_probs > np.random.rand(n_hidden)).astype(int)
v_recon_probs = sigmoid(np.dot(hidden_states, W.T))
v_recon = (v_recon_probs > 0.5).astype(int)

# --- Step control ---
step = st.session_state.get("step", 0)

if st.button("Next Step ➡️"):
    step = (step + 1) % 4  # 0=reset, 1=visible, 2=hidden, 3=recon
    st.session_state["step"] = step

# --- Visualization ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.axis("off")

# Positions
visible_y = 0
hidden_y = 1
x_visible = np.linspace(0, 1, n_visible)
x_hidden = np.linspace(0, 1, n_hidden)

# Draw connections (faded if before step 2)
for i in range(n_visible):
    for j in range(n_hidden):
        weight = W[i, j]
        color = "blue" if weight > 0 else "red"
        alpha = min(1, abs(weight))
        if step < 2: alpha *= 0.2  # fade before hidden activation
        ax.plot([x_visible[i], x_hidden[j]], [visible_y, hidden_y],
                color=color, alpha=alpha)

# Draw visible nodes
for i in range(n_visible):
    if step >= 1:  # show input
        color = "green" if v[i] == 1 else "gray"
    else:
        color = "lightgray"
    ax.scatter(x_visible[i], visible_y, s=500, c=color, edgecolors="black", zorder=3)
    ax.text(x_visible[i], visible_y - 0.05, f"v{i+1}", ha="center")

# Draw hidden nodes
for j in range(n_hidden):
    if step >= 2:  # activate hidden layer
        intensity = hidden_probs[j]
        color = plt.cm.Blues(intensity)
    else:
        color = "lightgray"
    ax.scatter(x_hidden[j], hidden_y, s=500, c=color, edgecolors="black", zorder=3)
    ax.text(x_hidden[j], hidden_y + 0.05, f"h{j+1}", ha="center")

ax.set_title(f"RBM Step {step}: " +
             ["Reset", "Input Visible Units", "Hidden Activation", "Reconstruction"][step])

st.pyplot(fig)

# Show reconstruction at final step
if step == 3:
    st.subheader("Step 3: Reconstruction")
    cols2 = st.columns(n_visible)
    for i in range(n_visible):
        color = "🟩" if v_recon[i] == 1 else "⬛"
        cols2[i].markdown(f"**v{i+1}** {color}")
