# Goal:
This project's goal is to create a NanoGPT with an embedding layer and a neural network.

# What I've learned:
- With the creation of this model I've learned how to split the training dataset and validation dataset,
- The various types of tensor in torch,
- Why the initial loss (~4.9) is above the uniform baseline (ln(65) ≈ 4.17), because random initialization cause over-confidence, but wrong predictions, which cross-entropy punishes harder than uniform guessing.
- The multi-head attention, wich with linear operation can give the full contest at the llm and the feedforward can weight the single token,
- The GPT model can make under 1.8 of loss and win against the simple bigram model.
