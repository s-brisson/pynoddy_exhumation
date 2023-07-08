# pynoddy_exhumation

This code is intended to calculate the average exhumation from multiple stochastic geokinematic models made in Noddy.


#### Output Files 
##### Parameters

Parameters fed to noddy to produce this specific model. They're modified in each iteration randomly. Each draw as a new set of parameters. They are uniquely tagged with a random string and the iterative number of the draw.

Example of the paramters output file
`data/outputs/bregenz/model_param/yyyymmdd_HHMM/params_<unique_label>.csv`
|Event | New Dip| New Dip Direction| New Pitch| New Slip| New Amplitude | New X | New Z| NDraw |
|----|-----| ------ | ----| -----|-----|-----| -----|----|
|    |     |        |     |      |     |     |      |    |

---
##### Model Coordinates
The coordinates of the whole model. For each draw there is one set of coordinates that represent the model itself. At the end of the simulation the NDRAW sets are concateated in a single pickle file with the following convetion

`data/outputs/bregenz/model_coords/yyyymmdd_HHMM/coords_<unique_label>.npy`

| N_Sample | X    | Y      | Z    | Exhumation | NDraw |
|----------|------|--------|------|------------|-------|
| sono     | una  | grande | alta | bol        | uda   |
---
##### Model Blocks
The block matrix that tells the rock type contained in each voxel. It is important for calculating the entropy and for visualizing the models. For each draw there is one block that represents the model itself. At the end of the simulation the NDRAW blocks are concateated in a single pickle file with the following convetion

`data/outputs/bregenz/model_block/yyyymmdd_HHMM/blocks_<unique_label>.npy`

The output file is a 4-index matrix 
```
[draw number, x voxel index, y voxel index, z voxel index]
```
which contains integers specifying the lithology ID
---
##### Model Scores
The scores of the model describing how well the model fits the input data. For each draw there is one score (i.e. a number from 1 to 10). A simple pandas dataframe file keeps track of the scores:

`data/outputs/bregenz/model_score/yyyymmdd_HHMM/scores_<unique_label>.csv`

| Draw  Numer | Score | 
|----------|------|
| 1     | Mas que un panch  |
---
##### Model Sample
The observations are input to the simulation as a pandas dataframe containing the coordinates of the sample, the ID of the sample and the group. During the simulation a new column is added to the dataframe adding +1 every time the observation matches the model.

`data/outputs/bregenz/model_score/yyyymmdd_HHMM/scores_<unique_label>.csv`


| I don't  | know | 
|----------|------|
|  |  |