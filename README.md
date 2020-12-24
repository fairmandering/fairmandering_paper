# gerrypy
Code associated with Fairmandering: A column generation heuristic for fairness-optimized political districting

## Setup
### Code setup
We use a number of dependencies that are best installed using the anaconda package manager. You can install conda [here](https://www.anaconda.com/products/individual). We also use Gurobi, which requires a license to use. You can download a free academic license [here](https://www.gurobi.com/downloads/end-user-license-agreement-academic/)
```
git clone https://github.com/fairmandering/fairmandering_paper.git
conda env create -f environment.yml
conda active gerry
python setup.py develop 
```
### Data setup
All of our raw data, processed data, generated columns, and optimization results can be found in our Box [data repository] (note, this amounts to 45GB of materials). To reproduce paper figures, all data from the box `Data` folder should be place in the `data` directory and the `Results (Columns)` folder should be placed in a parallel directory to the `data` in `gerrypy` and called `results`. All experiment files contain the column generation configs used by `gerrypy.optimize.generate.ColumnGenerator` to replicate the experiment.

To rerun our data collection and preprocessing pipeline, see `gerrypy/data/get.py`. This downloads census tract shapefiles and demographic statistics from the census server. Populate the `CENSUS_API_KEY` in `gerrypy.constants` using a census API key requested from [here](https://api.census.gov/data/key_signup.html). The last step of the processing pipeline requires manually downloaded election data that we gathered from [MGGG states](https://github.com/mggg-states), the Voting and Election Science Team (VEST) for [2016](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NH5S2I&version=41.0) and [2018](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UBKYRU), the [Harvard Election Data Archive](https://dataverse.harvard.edu/dataverse/eda), and the MIT Election Lab (MEDSL) presidential county [returns](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ). We compiled all of these files in the `precincts` folder in the `data` folder within box.


## Package Overview
The package is organized as follows.
### Data
The `data` subpackage contains all code required for downloading and preprocessing data to form the input to the generation and optimization routines. `get.py` is the main pipeline script. Additional elections can be integrated into the pipeline by making the appropriate additions to `precinct_state_wrappers.py`.

### Optimization
We use a 2 stage optimization approach, the first stage generates a large district ensemble which can be composed into an exponential number of district plans. With these districts as columns of the constraint matrix of the master problem, where the optimal solution to the master problems identifies the `k` optimal districts. These algorithms can be found in the `optimize` subpackage.

#### Column (district) generation

The `ColumnGenerator` class in `generate.py` is responsible for generating districts and recording various diagnostic metrics to be analyzed after generation. This object runs the stochastic hierarchical partitioning (SHP) algorithm and maintains the queue and tree data structures of the algorithm. These data structures are populated with `SHPNode`s from `tree.py` that keep track of data to reconstruct the sample tree. Within each partition routine, the `ColumnGenerator` uses one of the seed selection methods found in `center_selection.py` to choose the seeds for the partition. Each `ColumnGenerator` object requires a configuration dictionary to set all hyperparameters. These configs are saved by all experiment pipelines with the tree and other generation metadata.

#### District metrics
In between generation and optimization we use the `analyze` subpackage to study the properties of the ensemble, and aggregate tract metrics to compute district metrics for all districts in the ensemble. The `districts.py` module contains most of the relevant functions to do the aggregations, compute ensemble metrics, and construct the block district matrix for the master selection problem.

#### Fairness master selection problem
Using the block district matrix and `district_df` yielded by the generation ensemble, we use the `master.py` to shard the tree, and construct a Gurobi model for each root partition column space.

### Analysis and Experimentation
To perform our end-to-end experiments we have a number of pipelines to automate running experiment trials for different configurations and/or states. In the `experiments` subpackage we use the `generation.py` module to perform our SHP tuning experiments to test different SHP configs and PDP parameters. The `allstates.py` module runs a full ensemble and optimization pipeline for all multi-district states, and saves all intermediate results. The optimization pipeline it runs is the `pnas.py` module contained in `pipelines` which in addition to solving all master selection problems, uses the dynamic programming formulation to compute ensemble ranges of partisan advantage and compactness, and prunes the sample tree to make full enumeration tractable to compute ensemble distributional metrics.

Lastly, to construct the figures in the paper, we use the modules in the `paper.pnas` subpackage. `generate_pnas_paper` is the main script for generating all tables and figures and calls helper functions and figure generating functions in `all_states.py` and `algorithm_configuration.py`.



