# fair_regression_reduction_coreset-master


The repo contains the code for creating coreset for fair-regression problem -- [On Coresets for Fair Regression](https://drive.google.com/file/d/1z0oUWKc2wwzlSypOGdKs2DKACTGmKVNA/view) -- [Accepted at SubsetML-ICML'21](https://sites.google.com/view/icml-2021-subsetml/accepted-papers).

The addition is the code for creating the *fair* coreset and providing the same to the main fair-regression-reduction code --[repo here](https://github.com/steven7woo/fair_regression_reduction)

To run the coreset based code:

```
python3 run_exp_coreset.py r_Value epsilon_value dataset_name dataset_size
```
