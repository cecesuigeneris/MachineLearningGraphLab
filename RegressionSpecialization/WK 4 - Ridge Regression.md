# Ridge Regression

## Key Concepts:
* Overfitting (number of observations - few: easy to overfit; many - hard to overfit)
  
* Total cost = measure of fit + measure of magnitude of regression coefficients
  > measure of fit = RSS(w)
  > measure of magnitude = sum.absolute value of w (L1 norm); or sum.squares of w (L2 norm)

* Ridge objectives
  > total cost = RSS(w) + λ||w||2^2
  > λ -- tuning parameter/penalty strength, is to control model complexity and balance of fit and magnitude
      - λ=0, reduce to RSS(w), old solution, w (least sqaure)
      - λ=∞ , (1) w hat =/=0, total cost = ∞ ; (2) w hat = 0, cost = RSS(0)

* Bias-variance tradeoff
  > large λ, high bias, low variance
  > small λ, low bias, high variance (e.g. standard least sqaure RSS fit of high order polynomial for λ=0)

* K-fold cross validation (k = 5 or 10 typically): all subset of observation set as validation set then average them
  > best approx. occurs for validation sets of size 1 (K=N): leave-out-out cross validation
