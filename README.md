# Goal:
This project's goal is to create a NanoGPT with an embedding layer and a neural network.

# What I've learned:
-With the creation of this model I've learned how to split the training dataset and validation dataset,
-The various types of tensor in torch,
-Why the initial loss (~4.9) is above the uniform baseline (ln(65) ≈ 4.17): because random initialization makes confident, but wrong predictions, which cross-entropy punishes harder than uniform guessing.