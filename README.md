## Deep Learning Practice

**Training parameters**
- targetErrorDelta = 0.001F;
- LEARNING_RATE = 0.1F;

**Results**

Network with **1** hidden layer of **30** neurons took **3** epochs to train
Correct: **9406** out of **10000**

Network with **2** hidden layer of **30** neurons took **5** epochs to train
Correct: **9041** out of **10000**

Training with **1** hidden layer of **100** neurons took **3** epochs to train
Correct: **9639** out of **10000**

**Conclusions**

- deeper network took longer to train and performed worse
- using more neurons, but not lot layers, improved accuracy while took the same time to train


**Training parameters**
- targetErrorDelta = 0.0001F;
- LEARNING_RATE = 0.01F;

**Results**

Training with **1** hidden layer of **100** neurons took **16** epochs to train
Correct: **9673** out of **10000**

**Conclusions**

- Slowly learning network could outperform the fast-learning one after 13 epochs
- Slower learning is slow, but is slightly more accurate
- It's difficult to say whether the added accuracy is worth the extra time spent learning
- Even the slowest and most accurate network could not get my number 7 right
