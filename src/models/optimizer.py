import torch
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

class NoamOptimizer(BaseOptimizer):
    def __init__(self, model_size, factor, warmup_steps, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'step': self._step,
            'rate': self._rate
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._step = state_dict['step']
        self._rate = state_dict['rate']

def create_noam_optimizer(model, model_size=256, factor=1.0, warmup_steps=4000):
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    return NoamOptimizer(
        model_size=model_size,
        factor=factor,
        warmup_steps=warmup_steps,
        optimizer=optimizer
    )
    

def create_cnn_gru_optimizer(model, learning_rate=0.001):
    """
    Create an optimizer for the CNN+GRU model.
    
    Args:
        model: The neural network model
        learning_rate: Initial learning rate
    
    Returns:
        optimizer: The configured optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer

def create_cnn_gru_scheduler(optimizer, patience=5, factor=0.5, min_lr=1e-6):
    """
    Create a learning rate scheduler for CNN+GRU model.
    
    Args:
        optimizer: The optimizer
        patience: Number of epochs with no improvement after which learning rate will be reduced
        factor: Factor by which the learning rate will be reduced
        min_lr: A lower bound on the learning rate
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=patience,
        factor=factor,
        min_lr=min_lr,
        verbose=True
    )
    return scheduler