from torch import nn
import copy


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.model_weights = copy.deepcopy(model.state_dict())
        self.ema_weights = copy.deepcopy(model.state_dict())
        self.decay = decay
    
    def update(self):
        for name, param in self.model.state_dict().items():
            if param.dtype.is_floating_point:
                self.ema_weights[name] = self.decay * self.ema_weights[name] + (1 - self.decay) * param
            else:
                self.ema_weights[name] = param
            
    def apply(self):
        self.model_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_weights)
        
    def restore(self):
        self.model.load_state_dict(self.model_weights)