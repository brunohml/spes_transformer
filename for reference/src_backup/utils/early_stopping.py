import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience, val_interval, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Number of validation checks to wait
            val_interval (int): Number of epochs between validation checks
            verbose (bool): Whether to print improvement messages
            delta (float): Minimum required improvement in loss to be considered "better"
            path (str): Where to save the best model checkpoint
        """
        self.patience = int(patience)
        self.val_interval = int(val_interval)
        self.epochs_patience = self.patience * self.val_interval
        self.verbose = verbose
        self.counter = 0          # Counts epochs since last improvement
        self.best_score = None    # Best score seen so far
        self.early_stop = False   # Flag to indicate if training should stop
        self.val_loss_min = np.Inf  # Initialize best loss as infinity
        self.delta = float(delta)        # Minimum improvement threshold
        self.path = path          # Checkpoint save location

    def __call__(self, val_loss, model):
        """
        Main method called after each validation epoch
        Args:
            val_loss: Current validation loss
            model: Current model state
        """
        # Convert loss to score (negative since we want to minimize loss)
        score = -val_loss

        # First epoch - initialize best score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        # If we didn't improve enough (considering delta threshold)
        elif score < self.best_score + self.delta:
            self.counter += 1  # Increment patience counter
        
        # If we did improve enough
        else:
            self.best_score = score  # Update best score
            self.save_checkpoint(val_loss, model)  # Save model
            self.counter = 0  # Reset patience counter

        # Print status and check for early stopping after counter is updated
        if self.verbose:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} checks '
                  f'({self.counter * self.val_interval} out of {self.epochs_patience} epochs)')
        
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss