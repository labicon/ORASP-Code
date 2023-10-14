# Optimal Robotic Assembly Sequence Planning Code
Code for the paper: "Optimal Robotic Assembly Sequence Planning: A Sequential Decision-Making Approach"

## Installation
You can recreate the python environment used to run these files via:
```
conda create --name <env> --file requirements.txt
```

Similiarily, using a standard Python 3 installation, you can also use:
```
python3 -m venv env
source env/bin/activate
pip install -r pipRequirements.txt
```


## Running the Code
A few example scenarios are provided inside the Python Juypter Notebooks under the "Scenario Initialization" sections. 
The `GEAP` file is for the Graph-Exploration Assembly Planners (GEAPs) discussed in our "Methods" section, and the `DQN` file holds the Learning-Based methods.

