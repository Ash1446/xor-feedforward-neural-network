import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from xor_nn import XORNet, X, y, plot_boundary

st.set_page_config(page_title="XOR Solver", page_icon="🧠")
st.title("XOR Solver - Feedforward Neural Network")
st.write("A 2-2-1 network trained from scratch with backpropagation.")

@st.cache_resource
def train_model():
    net = XORNet(hidden=2, act="sigmoid", lr=0.5, seed=1)
    losses = net.train(X, y, epochs=10000)
    return net, losses

net, losses = train_model()

col1, col2 = st.columns(2)
a = col1.selectbox("Input 1", [0, 1])
b = col2.selectbox("Input 2", [0, 1])

p = net.forward(np.array([[a, b]], dtype=float))[0, 0]
st.success(f"Output probability: {p:.3f} → XOR = {int(p > 0.5)}")

st.subheader("Decision boundary")
fig, ax = plt.subplots()
plot_boundary(net, "Decision boundary (2-2-1, sigmoid)", ax)
st.pyplot(fig)

st.subheader("Training loss")
st.line_chart(losses)
