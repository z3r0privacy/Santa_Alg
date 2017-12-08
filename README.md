# Installation

Install the virtual environment including packages with

```
make install
```

# Run

To run a specific method, select it by providing its name for the MET variable,
e.g.

```
make MET=sim
```

# Data/solution exploration

Start an interactive python shell which loads `explore.py`, containing common
imports for convenience, loading of data, and visualization/analysis methods:

```
make explore
```

# Develop

Modify existing methods in `methods/` or add new methods by deriving from the
`Method` class in any python file in the directory.

When new dependencies are added, update `requirements.txt` by calling
`make freeze`.

