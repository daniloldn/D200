# D200
Coursework for a graduate-level Machine Learning in Economics module. Each problem set lives under `Problem_set_*/` with companion models in `Problem_set_*/Notebook/models/` and narrative code in the corresponding notebook.

## Problem Set 1 Highlights
- **Autograd warm-up:** Used PyTorch tensors to build and differentiate simple scalar and multivariate functions (see `problem_set_1.ipynb`, Cells 4–11) to cement intuition for gradients.
- **Linear regression from scratch:** Implemented `LinearRegression` in [Problem_set_1/Notebook/models/linear.py](Problem_set_1/Notebook/models/linear.py) and trained it on synthetic data using SGD, logging convergence diagnostics and comparing against the analytic OLS solution.
- **Optimizer comparison:** Tracked the loss trajectories for SGD vs Adam on the same regression problem to illustrate optimizer behavior differences, with Plotly visualizations embedded in the notebook (Cells 20–28).
- **MNIST MLP classifier:** Built a fully-connected network in [Problem_set_1/Notebook/models/mlp.py](Problem_set_1/Notebook/models/mlp.py) (Flatten → 256 → 128 → 10) and trained it on normalized MNIST digits using mini-batch Adam, capturing loss/accuracy curves plus qualitative evaluation on held-out digits.

