import logging
import subprocess
from pathlib import Path
import os
import sys
import time
import pickle
import json
import matplotlib as plt
import numpy as np

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from utils.early_stopping import EarlyStopping
from datasets.data import DynamicSPESData, data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer

def setup_logging(config):
    """Setup logging to both console and file"""
    # First, reset the logging configuration
    logging.getLogger().handlers = []
    logging.root.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')
    
    # Get logger and ensure it's clean
    logger = logging.getLogger('__main__')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    
    # Setup file handler
    log_file = os.path.join(config['output_dir'], "console_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Disable logging for other loggers
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
        if log_name != '__main__':
            if isinstance(log_obj, logging.Logger):
                log_obj.propagate = False
                log_obj.handlers = []
    
    return logger

def main(config):
    logger = setup_logging(config)

    metrics_dict = {
        'train_loss': [],
        'val_loss': [] 
    }

# only add accuracy metrics if it's a classification task
    if config['task'] == 'dynamic_classification':
        metrics_dict.update({
            'train_accuracy': [],
            'val_accuracy': []
        })

    total_epoch_time = 0
    total_eval_time = 0
    total_start_time = time.time()

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Prepare for dynamic loading
    data_class = data_factory[config['data_class']]
    train_data = data_class(config['data_dir'], pattern=config['pattern'], n_procs=config['n_proc'], limit_size=config['limit_size'], config=config)
    val_data = data_class(config['val_data_dir'], pattern=config['pattern'], n_procs=config['n_proc'], limit_size=config['limit_size'], config=config)
    
    logger.info(f"Training data has {len(train_data.all_IDs)} examples available")
    logger.info(f"Validation data has {len(val_data.all_IDs)} examples available")
    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, train_data) # just needs shape of train_data, so will be applicable to val_data

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # TODO: implement gradual unfreezing
    
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer
    # Set up L2 regularization
    if config.get('global_reg', False):
        # Apply L2 reg through optimizer's weight decay
        weight_decay = float(config.get('l2_reg', 0.0))
        output_reg = None
    else:
        # Apply L2 reg in loss calculation
        weight_decay = 0
        output_reg = torch.tensor(float(config.get('l2_reg', 0.0)), device=device)
    
    # Create optimizer with appropriate weight decay
    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0 # current step index of "lr_step"
    lr = config['lr'] # current learning step

    # Load model and optimizer state
    if args.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                        config['change_output'],
                                                        config['lr'], config['lr_step'], config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)

    # TODO: implement test only

    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_data.all_IDs)
    train_dataset = dataset_class(train_data, train_data.all_IDs)

    # TODO: implement samplers if balanced sampling is desired

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=model.max_len)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=model.max_len)
    )

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                           print_interval=config['print_interval'], console=config['console'])
    
    val_evaluator = runner_class(model, val_loader, device, loss_module, optimizer=None, 
                                 l2_reg=config['l2_reg'] if 'l2_reg' in config else None,
                                 print_interval=config['print_interval'],
                                 console=config['console'])

    tensorboard_writer = SummaryWriter(log_dir=config['tensorboard_dir'])

    # Initialize best metrics and values
    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16
    metrics = []
    best_metrics = {}

    # Evaluate on validation set before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer,config, best_metrics, best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], val_interval=config['val_interval'], 
                                   delta=config['early_stopping_delta'], verbose=True,
                                   path=os.path.join(config['save_dir'], 'early_stopping_checkpoint.pt'))
    
    logger.info("Starting training ...")
    for epoch in tqdm(range(start_epoch + 1, config['epochs'] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last' # save all or last model
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch) # dictionary to aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            print_str += '{}: {:.4f} '.format(k, v)
        logger.info(print_str)
        logger.info('Epoch runtime: {:.4f} seconds'.format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime

        # Store training metrics before validation check
        metrics_dict['train_loss'].append(aggr_metrics_train['loss'])

        # Store accuracy for plotting if it's a classification task
        if 'accuracy' in aggr_metrics_train:
            metrics_dict['train_accuracy'].append(aggr_metrics_train['accuracy'])

        # Then do validation check and storage
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

            # Store validation metrics for plotting
            metrics_dict['val_loss'].append(aggr_metrics_val['loss'])
            if 'accuracy' in aggr_metrics_val:
                metrics_dict['val_accuracy'].append(aggr_metrics_val['accuracy'])

            # Check early stopping conditions
            early_stopping(aggr_metrics_val['loss'], model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered - training stopped")
                break
        else:
            # For epochs where we don't validate, append NaN
            metrics_dict['val_loss'].append(float('nan'))
            if 'accuracy' in metrics_dict:
                metrics_dict['val_accuracy'].append(float('nan'))

        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor']
            if lr_step < len(config['lr_step']) - 1: # so index doesn't get out of range
                lr_step += 1
            logger.info('Learning rate updated to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()
    
    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    # Generate and save the plots
    print("\nFinal metrics shapes:")
    print(f"Train loss points: {len(metrics_dict['train_loss'])}")
    print(f"Val loss points: {len(metrics_dict['val_loss'])}")
    print(f"Validation losses: {[x for x in metrics_dict['val_loss'] if not np.isnan(x)]}")

    # Get model directory name from output_dir
    model_dir = os.path.basename(config["output_dir"])

    # Plot the metrics with all information
    utils.plot_metrics(
        config['output_dir'], 
        metrics_dict,
        model_dir=model_dir,
        comment=config.get('comment')
    )

    # After training loop and metrics export
    if config['task'] == 'imputation' or config['task'] == 'dynamic_imputation':
        logger.info('Training completed. Running visualization script...')
        
        # Construct paths
        model_path = Path(config['output_dir'])
        vis_output_path = model_path / 'imputation_results.png'
        
        # Get path to visualization script relative to dynamic_main.py
        vis_script_path = Path(__file__).parent.parent / 'visualize_imputation.py'
        
        # Run visualization script as subprocess
        try:
            subprocess.run([
                sys.executable,  # Current Python interpreter
                str(vis_script_path),
                '--model_path', str(model_path),
                '--output_path', str(vis_output_path)
            ], check=True)
            logger.info(f'Visualization saved to {vis_output_path}')
        except subprocess.CalledProcessError as e:
            logger.error(f'Visualization script failed with error: {e}')
        except Exception as e:
            logger.error(f'Unexpected error running visualization: {e}')
    else:
        logger.info('Training completed.')

    # Return best value as before
    return best_value

if __name__ == '__main__':
    args = Options().parse() # 'argsparse' object
    config = setup(args) # configuration dictionary
    main(config)