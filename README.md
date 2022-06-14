<div align="center">    
 
# Estimating the Probabilities of Causation with Deep Monotonic Twin Networks




<!--  
Conference   
-->   
</div>
 
## Description   
This repo contains the supporting code for the paper: "Estimating Categorical Counterfactuals via Deep Twin Networks"
by : Athanasios Vlontzos, Bernhard Kainz, Ciaran Gilligan-Lee



## How to run   
First, install dependencies   
```bash
# install project   
  
pip install -r requirements.txt
 ```   
 ### Synthetic Experiments 
 - Unconfounded
 
 Create the data 
 ```bash
cd ./data
python synthetic_dataset.py --u_distribution U_DISTRIBUTION


```
Run the model 
```bash
cd ..
cd ./Synthetic_Train
python train_default_loop.py  --ARGS
```
Calculate the Probabilities of Causation
```bash
python calc_probs.py  --ARGS
```

## Data 
For the Twins data please use 
```
https://github.com/jsyoon0823/GANITE/blob/master/data/Twin_data.csv.gz
```
For the Kenyan Water Dataset 
```
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910/DVN/28063
```
## Hparams 

### Synthetic wt confounders:
```python
    parser.add_argument('--confounders',default=['Z'])

    parser.add_argument('--u_distribution', default='normal')
    parser.add_argument('--z_distribution', default='uniform')
    parser.add_argument('--x_distribution', default='bernouli')
    parser.add_argument('--p_1', type=float, default=0.05)
    parser.add_argument('--p_2', type=float, default=0.7)
    parser.add_argument('--p_3', type=float, default=0.2)

 # Model Hparams
    parser.add_argument('--lattice_sizes', default=[3, 3])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=3)
    parser.add_argument('--calib_units', type=int, default=3)
    parser.add_argument('--z_calib_units', type=int, default=3)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monot_opt', default=1)
    parser.add_argument('--concats', type=bool, default=False)

    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--weighted_loss',default=False)
    parser.add_argument('--weight_1',type=float,default=1)
    parser.add_argument('--weight_2',type=float,default=1)
    parser.add_argument('--multiple_confounders', default=True, help='split confounders')

```

### Synthetic unconfounded

```python
     parser.add_argument('--u_distribution', default='uniform', help='normal, uniform')

    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[2, 3, 2, 2, 3])
    parser.add_argument('--lattice_monotonicities', default=['increasing', 'increasing'])
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--hidden_dims', default=1)
    parser.add_argument('--calib_units', default=2)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')

```

### Kenyan Water

```python
    
    parser.add_argument('--u_distribution', default='normal')
    parser.add_argument('--confounders', default=['base_age', 'splnecmpn_base', 'latrine_density_base'])
    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[4, 4])
    parser.add_argument('--lattice_monotonicities', default=['increasing', 'increasing'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=4)
    parser.add_argument('--calib_units', type=int, default=4)
    parser.add_argument('--z_calib_units', type=int, default=4)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monot_opt', default=2)
    parser.add_argument('--concats', type=bool, default=False)

    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--multiple_confounders', default=True, help='split confounders')

```

### Twins Dataset

```python
    
    parser.add_argument('--confounders', default='all')


    parser.add_argument('--u_distribution', default='normal')
    # Model Hparams
    parser.add_argument('--lattice_sizes', default=[3,3])
    parser.add_argument('--lattice_monotonicities', default=['increasing', 'increasing'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lattice_units', type=int, default=1)  # 1 or 2
    parser.add_argument('--hidden_dims', type=int, default=3)
    parser.add_argument('--calib_units', type=int, default=3)
    parser.add_argument('--z_calib_units', type=int, default=3)
    parser.add_argument('--layer', default='lattice')
    parser.add_argument('--uy_layer', default='none')
    parser.add_argument('--z_layer', default='none')
    parser.add_argument('--uy_monotonicity', default='none')
    parser.add_argument('--z_monotonicity', default='none')
    parser.add_argument('--z_monot_opt', default=1)
    parser.add_argument('--concats', type=bool, default=False)

    parser.add_argument('--end_activation', default='none')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--weighted_loss', default=False)
    parser.add_argument('--weight_1', type=float, default=1)
    parser.add_argument('--weight_2', type=float, default=1)
    parser.add_argument('--multiple_confounders', default=False, help='split confounders')
```

## Categorical Datasets 

### Stroke and Credit

- Download 
  - [International Stroke Trial Dataset](https://datashare.ed.ac.uk/handle/10283/124)
  - [German Credit Score Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

- Create the datasets by running the dataloader scripts 
- Run with default hparams 

## Citation

``` 
@article{vlontzos2021estimating,
  title={Estimating Categorical Counterfactuals via Deep Twin Networks},
  author={Vlontzos, Athanasios and Kainz, Bernhard and Gilligan-Lee, Ciaran M},
  journal={arXiv preprint arXiv:2109.01904},
  year={2021}
}
```
