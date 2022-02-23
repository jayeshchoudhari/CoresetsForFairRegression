# Coreset for Fair Regression


The repo contains the code for creating coreset for fair-regression problem -- [On Coresets for Fair Regression](https://drive.google.com/file/d/1z0oUWKc2wwzlSypOGdKs2DKACTGmKVNA/view) -- [Accepted at SubsetML-ICML'21](https://sites.google.com/view/icml-2021-subsetml/accepted-papers). The paper containing this work is also accepted at AISTATS 2022 and is titled "On Coresets for Fair Regression and Individually Fair Clustering". To access the code for Individual Fair Clustering the link is: https://github.com/jayeshchoudhari/CoresetIndividualFairness

The addition is the code for creating the *fair* coreset and providing the same to the main fair-regression-reduction code --[repo here](https://github.com/steven7woo/fair_regression_reduction). This preceding link is the code for the original paper on fair regression (http://proceedings.mlr.press/v97/agarwal19d/agarwal19d.pdf) and is made available through the link by the authors

To run the coreset based code:

```
python3 run_exp_coreset.py r_Value epsilon_value dataset_name dataset_size
```
