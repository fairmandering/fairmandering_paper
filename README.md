# gerrypy
Fixing gerrymandering by optimizing for fairness

## Package Overview
The package is organized as follows.
### Data collection
The column generation routine requires 3 data structures: An adjacency graph, a pairwise distance matrix, and a state dataframe containing the position and population of every geographic unit (precinct, census tract/block, county, etc.). The fairness master problem requires a vector of probabilities of party A winning the district in an arbitrary election. This requires gathering a historical statewide election data to determine the mean and standard deviation of political affiliation in that district.


Once the state shapefiles, census data, and raw voting data is collected we create the above data structures. `preprocess.py` brings everything together by calling the data matching routine, fitting a GP regression model to census demographics to compute a mean and covariance matrix, and creating the pairwise distance matrix and the state dataframe. `adjacency.py` is used to clean up an adjacency graph for the state census tracts. When using census tracts for demographic data and precincts for voter data, one must project the attributes of one unit on to the other. We use `data_matching.py` to infer the demographic attributes of precincts using a (population and overlap) weighted average over the overlapping tracts. 

### Optimization
We use a 2 stage optimization approach, the first stage generates a large number of districts which can be composed into an exponential number of district plans. With these districts as columns of the constraint matrix of the master problem, where the optimal solution to the master problems identifies the `k` optimal districts.
#### Column (district) generation

The `ColumnGenerator` class in `generate.py` is responsible for generating districts and recording various diagnostic metrics to be analyzed after generation. This object runs the stochastic hierarchical partitioning (SHP) algorithm and maintains the queue and tree data structures of the algorithm. These data structures are populated with `SHPNode`s from `tree.py` that keep track of data to reconstruct the sample tree. Within each partition routine, the `ColumnGenerator` uses one of the seed selection methods found in `seed.py` to choose the seeds for the partition.

#### Fairness master problem
Once the set of columns are generated, we need to transform the input for the master problem integer program. We use `districts.py` to build the block district matrix (the binary matrix with element A_ij = 1 when block i is in district j) and the district_df (a dataframe containing all relevant demographic and electoral statsitics aggregated by block). This data is then passed the master IP in `master.py` which returns a model.

#### Query tree
If there are no additional constraints (e.g. minimum competitiveness, majority-minority districts, compactness) and the objective is linear (unfairness, compactness, competitiveness) then sample tree admits an efficient optimal solution by dynamic programming. In this case we use `analyze\tree.py` to solve such problems.

### Analysis and Experimentation
A number of helper methods to compute metrics regarding specific plans or individual districts can be found in the `analyze` directory.

### Display and Visualization
The `app` folder in `analyze` contains a (currently buggy) webapp that does a nice job of visualizing how the column generation routine proceeds in a dynamic way. `viz.py` contains a few functions for static visualizations of adjacency graphs and district maps.

