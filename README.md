# Goal:
- Build a NanoGPT and study the evolution of the model.

# Research:
- Try with 55k parameters and 10M:
  This is the initial version of NanoGPT,it has all the tools of a NanoGPT (like multi-head attention, feedforward, ...), I've run this model for 15000 steps and it reached the limit of      1.8      loss on the validation set, with the current dataset this model is too simple and the model is capacity-limited.
  Then I've run the same architecture, scaled up to 10M parameters, this model memorizes the patterns and overfits, with a 1.55 in validation set but 1.15 in training set, then I've        updated    the model with a dropout method, which turns off a random percentage of neurons in training to prevent the unstable connections used for memorization, and with an amount of 20% the model            reached 1.499 loss on the validation set with 1.22 on the training set.
