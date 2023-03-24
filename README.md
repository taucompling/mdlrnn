# Minimum Description Length Recurrent Neural Networks

![license](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9-blue)
![license](https://img.shields.io/badge/license-GNU-green)
![code style](https://img.shields.io/badge/code_style-Black-black) 
[![arXiv](https://img.shields.io/badge/arXiv-2111.00600-b31b1b.svg)](https://arxiv.org/abs/2111.00600)


Code for [Minimum Description Length Recurrent Neural Networks](https://arxiv.org/abs/2111.00600) by Nur Lan, Michal Geyer, Emmanuel Chemla, and Roni Katzir.

Paper: https://arxiv.org/abs/2111.00600

<img src="assets/anbncn.png" width="390" style="margin: 15px 0 5px 0"> 

## Getting started
1. Install Python >= 3.7
2. `pip install -r requirements.txt`

### On Ubuntu, install:
```
$ apt-get install libsm6 libxext6 libxrender1 libffi-dev libopenmpi-dev
```

## Running simulations

```
$ python main.py --simulation <simulation_name> -n <number_of_islands>
```

For example, to run the `aⁿbⁿcⁿ` task using 16 island processes:
```
$ python main.py --simulation an_bn_cn -n 16
```

* All simulations are available in `simulations.py`

* Final and intermediate solutions are saved to the `networks` sub-directory, both as `pickle` and in visual `dot` format.


## Parallelization

Native Python multiprocessing is used by default. To use MPI, change `migration_channel` to `mpi` in `simulations.py`.

## Citing this work

```
@article{Lan-Geyer-Chemla-Katzir-MDLRNN-2022,
  title = {Minimum Description Length Recurrent Neural Networks},
  author = {Lan, Nur and Geyer, Michal and Chemla, Emmanuel and Katzir, Roni},
  year = {2022},
  month = jul,
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {10},
  pages = {785--799},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00489},
}
```