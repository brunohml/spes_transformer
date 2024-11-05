# spes_transformer
Multivariate timeseries transformer based on Zerveas et al., 2021

Changes so far:
- Added early stopping mechanism via utils, defined at command line via early_stopping_patience, early_stopping_delta, and val_interval
- Added train/val loss and accuracy plots
- Added sensitivity, specificity, and Youden index metrics in output log excel sheet
