# Deep Collaborative Filtering

#### 10/01/2019
  - Attempted to create a library of functions to use for recommender datasets.
  - All functions can be found in `interactions_data.py` and `interactions_model.py`.
  - `CLR.py` and `OneCycle.py` are taken from    [this repo](https://github.com/nachiket273/One_Cycle_Policy).
  - Was having trouble getting model to learn anything, could not figure out why.
  - See `movielens_model.ipynb` for my attempted usage of these libraries.
  - Switched to using Skorch, with immediate success, see `movielens_data.ipynb` for example.
