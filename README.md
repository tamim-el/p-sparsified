# Fast Kernel Methods for Generic Lipschitz Losses via p-Sparsified Sketches

## Summary

This Python package contains code to use $p$-sparsified sketches for kernel methods.

## Installation

Necessary Python packages:
- numpy
- scikit-learn
- scipy
- pandas
- matplotlib
- liac-arff
- scikit-multilearn
- seaborn

To install them, run the following command:
pip install -r requirements.txt

## Details

Data: contains
- The data set rf1 (Spyromitros-Xioufis et al., 2016)
- The data set rf2 (Spyromitros-Xioufis et al., 2016)
- The data set scm1d (Spyromitros-Xioufis et al., 2016)
- The data set scm20d (Spyromitros-Xioufis et al., 2016)

Methods: contains 7 files:
- Sketch.py contains Python classes to implement various types of sketching
- RFF.py contains a Python class to implement Gaussian Random Fourier Features
- ScalarModel.py contains a Python class to implement non-sketched and sketched kernel learning in the single output setting
- ScalarModelRFF.py contains a Python class to implement kernel learning using RFF in the single output setting
- ChoiceM.py contains functions to compute various choices of matrix M in the multiple output setting
- QuantileModel.py contains a Python class to implement non-sketched and sketched decomposable kernel learning for joint quantile prediction
- VectorialModel.py contains a Python class to implement non-sketched and sketched decomposable kernel learning for multi-output regression


Utils: contains 2 Python files:
- load_data.py where data loading functions are implemented
- create_df.py where a function to generate a dataframe for the plots generated by run_synthetic.py and run_synthetic_All.py is implemented

Plots: contains the plots after running the scripts

## Use

RUN python files:
- run_synthetic.py: reproduces the plots for the kappa-Huber or epsilon-SVR
  and p-SR or p-SG sketches if run respectively with:
  python run_synthetic.py k_huber/e_svr Rademacher/Gaussian
- run_synthetic_All.py: run run_synthetic.py in all above configurations with:
  python run_synthetic_All.py
- run_rf1.py: reproduces all results reported in the paper on rf1 dataset with:
  python run_rf1.py
- run_rf2.py: reproduces all results reported in the paper on rf2 dataset with:
  python run_rf2.py
- run_scm1d.py: reproduces all results reported in the paper on scm1d dataset with:
  python run_scm1d.py
- run_scm20d.py: reproduces all results reported in the paper on scm20d dataset with:
  python run_scm20d.py
  
To ignore warnings outputs, please use -W option. Example:
python -W ignore run_rf1.py

## Cite

If you use this code, please cite the corresponding work:

```bibtex
@article{elahmad2023fast,
  title={Fast Kernel Methods for Generic Lipschitz Losses via $p$-Sparsified Sketches},
  author={Tamim {El Ahmad} and Pierre Laforgue and Florence d'Alch{\'e}-Buc},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=ry2qgRqTOw},
}
```
