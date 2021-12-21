## Plan of Experiments [building]

#### Neuron Models

1. Neuron Classic Inner Product
2. Neuron HSGS
3. Neuron  Encoding Weights + HSGS Input

4. Neuron Classic Inner Product / + Bias
5. Neuron HSGS / + Bias
6. Neuron  Encoding Weights + HSGS Input / + Bias

#### Data Sets

1. Train Cross - Test Cross with noise  [1 to 3 noises]
2. Train X and Cross (0) vs Square (1) - Test X, Cross and Square [1 to 3 noises] 
3. Train X  and Square (0) and Cross (1)  - Test X, Cross and Square [1 to 3 noises] 
4. Train X (0) vs Square and Cross (1) - Test X, Cross and Square [1 to 3 noises] 
5. Train X  (0) and Square (1) and Cross (2)  - Test X, Cross and Square [1 to 3 noises] 
6. Train horizontal lines - Test horizontal lines with noise [1 to 3 noises]
7. Train horizontal and vertical lines (0) - Test horizontal and vertical lines with noise [1 to 3 noises]

#### Metrics to capture during execution

1. Mean Accuracy (execute experiment X? times)
2. Time to convergence (reach 0.9 accuracy ?)

#### Metrics for evaluation

1. 95% confidence interval for accuracy
2. Friedman and Kolmogorov Smirnov tests between models
