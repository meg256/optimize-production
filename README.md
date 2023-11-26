# optimize-production
Interactive Streamlit dashboard showing workflow for manufacturing process optimization.

User uploads numerical csv data and selects response variable (test dataset provided in /src is a simulated process dataset obtained from Kaggle).

Evaluates degrees 1-5 and returns model with best degree based on RMSE

Trains best-degree polynomial regression model and determines global maximum with constrained optimization (COBYLA/scipy minimize).

Constraints are automatically generated from features of dataset - lower bound is 20% less than feature minimum, upper bound is 20% mroe than feature maximum.


