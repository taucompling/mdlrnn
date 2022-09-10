# Minimum Description Length Recurrent Neural Networks

Code for [Minimum Description Length Recurrent Neural Networks](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00489/112499/Minimum-Description-Length-Recurrent-Neural) by Nur Lan, Michal Geyer, Emmanuel Chemla, and Roni Katzir.

Paper: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00489/

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
