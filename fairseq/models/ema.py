import torch
from torch import nn

class EMA:
    def __init__(self, decay=0.9999):
        self.decay = decay
        self._averages = {} # type tensor

    def assign(self, name, param):
        self._averages[name] = param.detach() #requires_grad=False

    def step(self, name, param, updates):
        decay = min(self.decay, (1 + updates)/(10 + updates)) # weak the first #00 batches
        self._averages[name] = decay * self._averages[name] + (1-decay) * param.data
        
    def tensors_to_restore(self):
        return self._averages

def ema_init(ema, model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.assign(name, param)

def ema_step(ema, model, updates):
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.step(name, param, updates)

def ema_restore(ema, model):
    old_data = {}
    averages = ema.tensors_to_restore()
    for name, param in model.named_parameters():
        if name in averages:
            old_data[name] = param.data
            param.data = averages[name]
    return old_data

def ema_reverse(ema, model, old_data):
    for name, param in model.named_parameters():
        if name in old_data:
            param.data = old_data[name]
