# pynoddy_exhumation

This code is intended to calculate the average exhumation from multiple stochastic geokinematic models made in Noddy.


#### Output Files 
##### Parameters

Parameters fed to noddy to produce this specific model. They're modified in each iteration randomly. Each draw as a new set of parameters. They are uniquely tagged with a random string and the iterative number of the draw.
Example of the paramters output file
```
data/outputs/bregenz/model_param/yyyymmdd_HHMM/all_params.csv

all_params.csv
'Event', 'New Dip', 'New Dip Direction', 'New Pitch', 'New Slip', 'New Amplitude', 'New X', 'New Z', 'NDraw', 'UniqueLabel'
```

##### Model Coordinates
The coordinates of the whole model. For each draw there is one set of coordinates that represent the model itself. At the end of the simulation the NDRAW sets are concateated in a single pickle file with the following convetion

```
data/outputs/bregenz/model_coords/yyyymmdd_HHMM/coords_<unique_label>.npy

```

| N_Sample | X    | Y      | Z    | Exhumation | NDraw |
|----------|------|--------|------|------------|-------|
| sono     | una  | grande | alta | bol 

Example of the paramters output file
```
data/outputs/bregenz/model_param/yyyymmdd_HHMM/all_params.csv

```
|Event | New Dip| New Dip Direction| New Pitch| New Slip| New Amplitude | New X | New Z| NDraw | UniqueLabel|
|----|-----| ------ | ----| -----|-----|-----| -----|----|-------|
|    |     |        |     |      |     |     |      |    |       |
