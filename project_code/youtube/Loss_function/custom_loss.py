from abc import ABC
from abc import abstractmethod
# static method
class Loss(ABC):
    # @abstractmethod
    # def __init__(self) -> None:
        # super().__init__()
        
    @abstractmethod
    def Alignment(self) -> None:
        """Define layers in ther model."""
        raise NotImplementedError
    
    @abstractmethod
    def Anisotropy(self) -> None:
        """Define layers in ther model."""
        raise NotImplementedError
    def Total_loss(self) -> None:
        """Define layers in ther model."""
        raise NotImplementedError




import torch
import torch.nn.functional as F

# Loss function 
class SimLoss(Loss): # what i want to similar
    def __init__(self,hi1:torch.tensor
                ,ht1:torch.tensor
                ,hi2:torch.tensor
                ,ht2:torch.tensor
                ):
        self.hi1 = hi1
        self.ht1 = ht1
        self.hi2 = hi2
        self.ht2 = ht2
        self.tau = 1/0.05
        self.batch_size = batch_size
        
    def Alignment(self):
        Ali1 = -(self.tau*F.cosine_similarity(self.hi1,self.ht1))
        Ali2 = -(self.tau*F.cosine_similarity(self.hi2,self.ht2))
        return (Ali1+Ali2)/self.batch_size
    
    def Anisotropy(self):
        anisotropy = torch.empty(batch_size)
        for i in range(batch_size):
            I1 = self.hi1[i]
            I2 = self.hi2[i]
            anisotropyj = 0
            for j in range(batch_size):
                if i!=j:
                    T1 = self.ht1[j]
                    T2 = self.ht2[j] 
                    anisotropyj+=F.cosine_similarity(I1,T1,dim=0)
                    anisotropyj+=F.cosine_similarity(I2,T2,dim=0)
                    # tau 값을 넣으면 너무 커져서 안됨 .. batch size에 따라서 조절해야할듯..        
            anisotropy[i] = anisotropyj/self.batch_size#*tau
        return anisotropy
    
    def Total_loss(self):
        alignment  = self.Alignment()
        anisotropy = self.Anisotropy()
        return torch.sum(alignment+anisotropy)
# batch_size = 64
# h1 = torch.randn(64,152)
# h2 = torch.randn(64,152)
# total_loss = SimLoss(h1,h2,h1,h2)
# total_loss.Alignment().shape
# total_loss.Total_loss()