# gerrypy
Fixing gerrymandering by optimizing for fairness

## Package Overview
The package is organized as follows.
### Data collection
The column generation routine requires 3 data structures: An adjacency graph, a pairwise distance matrix, and a state dataframe containing the position and population of every geographic unit (precinct, census tract/block, county, etc.). The fairness master problem requires a political affiliation mean vector and coveriance matrix between every geographic unit.

TODO: improve data download scripts
TODO: expand number of elections

Once the state shapefiles, census data, and raw voting data is collected we create the above data structures. `adjacency.py` creates an adjacency graph of a `GeoPandas.GeoDataFrame` using an interpolation algorithm (see module for detailed documentation). When using census tracts for demographic data and precincts for voter data, one must project the attributes of one unit on to the other. We use `data_matching.py` to infer the demographic attributes of precincts using a (population and overlap) weighted average over the overlapping tracts. `preprocess.py` brings everything together by calling the data matching routine, fitting a GP regression model to census demographics to compute a mean and covariance matrix, and creating the pairwise distance matrix and the state dataframe.

TODO: Build out better GP infra

### Column (district) generation

TODO: remove deprecated files
TODO: create baseline directory

The `ColumnGenerator` object in `partition.py` is responsible for generating districts and recording various diagnostic metrics to be analyzed after generation. This object runs the stochastic hierarchical partitioning (SHP) algorithm and maintains the queue and tree data structures of the algorithm. These data structures are populated with `SHPNode`s from `tree.py` that keep track of data to reconstruct the sample tree. Within each partition routine the `ColumnGenerator` uses one of the seed selection methods found in `seed.py` to choose the seeds for the partition.

TODO: seed selection module

### Fairness master problem
TODO: master module

### Analysis and Experimentation
TODO: diversity metric
TODO: seed selection comparison
TODO: hyperparameter tuning

### Display and Visualization

