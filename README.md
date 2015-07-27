# MultitaskDescriptor
Implementation of the Multitask Lasso Descriptor. To install the package download it and execute `python setup.py install`.

The package requires 'numpy', 'scikit-learn' and 'hyperopt'. It is also recommended to install the package [`spams`](http://spams-devel.gforge.inria.fr/) for the optimization.

The package contains only one class `MultitaskLassoDescriptor`. To train the model the method `optimize` must be called. The method for predicting is `predict`.
