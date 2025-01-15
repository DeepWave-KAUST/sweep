import torch
import geomloss

class MSE(torch.nn.Module):
    """Mean Squared Error(L2) loss function.
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, syn, obs):
        return (0.5*(syn - obs) ** 2).mean()
    
class RTM(torch.nn.Module):
    """Reverse Time Migration loss function.
    For the RTM, the adjoint source should be the the observed data, 
    so the loss function is the product of the synthetic and observed data.
    """
    def __init__(self):
        super(RTM, self).__init__()

    def forward(self, syn, obs):
        return syn*obs
    
class CosineSimilarity(torch.nn.Module):
    def __init__(self, axis=1):
        """Cosine similarity (also the Global norm, Normalied cross correlation) loss function.
        Args:
            axis (int, optional): The time axis. Defaults to 1.
        """
        self.axis = axis
        super(CosineSimilarity, self).__init__()

    def forward(self, syn, obs):
        return (1-torch.nn.functional.cosine_similarity(syn, obs, dim=self.axis)).mean()
    
class Sinkhorn(torch.nn.Module):
    def __init__(self, p=2, blur=0.01):
        """Sinkhorn loss function.
        Args:
            p (int, optional): The p-norm. Defaults to 2.
            blur (float, optional): The blur factor. Defaults to 0.01.
        """
        self.p = p
        self.blur = blur
        super(Sinkhorn, self).__init__()
        self.creteria = geomloss.SamplesLoss("sinkhorn", p=self.p, blur=self.blur, debias=False)

    def forward(self, syn, obs):
        return self.creteria(syn.squeeze(), obs.squeeze()).mean()