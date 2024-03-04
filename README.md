# Replication Code for the paper "Sentiment-Driven Speculation in Financial Markets with Heterogeneous Beliefs: a Machine Learning approach"

## Overview
The repository is divided in two main folders: Models and Estimation.
The modeling part is in Julia and can be used to replicate the figures that pertain to the theoretical model.
The estimation part is in Python and can replicate figures and tables relative to the estimation part.

## Models - Instructions

1. To replicate all the figures in the paper, you can simply run the two files "plot_figures_single_param.jl" and "plot_figures_multiple_params.jl". The figures are based on the data that are already present in the "Data" folder.
2. To change parameters and explore the models you can change the values in the "params.json" file. Then run "generate_data_single_param.jl" and "generate_data_multiple_params.jl". After running the two files you **will overwrite** the data in the "Data" folder. At that point you can generate new figures by repeating step 1.
3. Finally, if you want to explore or change the main functions, you can do this in the "models.jl" file.

## Estimation - Instructions
1. You can just run the files "estimation_functions.py" and "bootstrap_confidence_errors.py" to obtain the estimate in the table of the paper and the bootstrap estimates in the appendix.
This is done by using the data already present in the "Data" folder, and will generate two files "params_df.csv" and "bootstrap_confidence_errors" in the "Data" folder. 
2. You can explore or change the estimation database by using the notebook "estimation_db.ipynb". If you run the cell in which we save the estimation database, you **will overwrite** the data in the "Data" folder.
3. The Jupyter notebook "explore_window.ipynb" can be used to explore the effect of different windows on the estimation results and reproduces the figure in the appendix.