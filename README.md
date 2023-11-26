# optimize-production
Interactive Streamlit dashboard showing workflow for manufacturing process optimization.

User uploads numerical csv data and selects response variable (test dataset provided in /data is a simulated process dataset obtained from Kaggle).

Exploratory data info is shown.

Evaluates sklearn polynomial models with degrees 1-5 and returns "best" degree based on RMSE.

Trains best-degree polynomial regression model and plots residuals/fit.

Determines global maximum with constrained optimization (COBYLA/scipy minimize).

Constraints are automatically generated from features of dataset - lower bound is 20% less than feature minimum, upper bound is 20% mroe than feature maximum.


