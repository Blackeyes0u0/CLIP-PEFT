# <center> Vector Universe </center>
---

<center>

![GitHub 로고](README_image/0u0.png)

</center>

#### <center> JoongHyun Shin </center>

<br>

##### <right>#CLIP #LoRA #Embeddings #Youtube Multimodal </right>

---
# 1. Datasets

![Alt text](image-2.png)

#### youtube thumbnails data

$I^{(i)}$ : youtube thumbnail Image data $i$
$T^{(i)}$ : youtube Title data $i$

#### Example 

$I^{(i)}$ : ![Alt text](image-1.png)

$T^{(i)}$ : **Cutest Cats Compilation 2017 | Best Cute Cat Videos Ever**


---
# 2. Model & Loss Architecture

<!-- 
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png) -->

<!-- ![Alt text](image-5.png) -->
![Alt text](image-7.png)

위 모델 아키텍쳐를 보면 Latent space 상에서 이미지 임베딩과 텍스트 임베딩의 거리를 가깝게 하는것을 alignment, 멀게하는것을 Uniform라고 정의하였습니다.

- LoRA 설명
#### https://velog.io/@blackeyes0u0/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-LoRA-Low-Rank-Adaptation-of-Large-Language-Models

- model 설명
### https://velog.io/@blackeyes0u0/youtube-CLIP-LoRA-SimCSE-%EA%B2%B0%EA%B3%BC

---


# 3. Objective function

$$
h_i = f(x_i)
$$

$$
\ell_i=-\log \frac{e^{\operatorname{sim}\mathbf{h}_i^{z_i}, \mathbf{h}_i^{z_i^{\prime}} / \tau}}{\sum_{j=1}^N e^{\operatorname{sim}\mathbf{h}_i^{z_i}, \mathbf{h}_j^{z_j^{\prime}} / \tau}}
$$

$i$ 번째 데이터와 $N$개의 batch_size pair 대해서 위와 같이 표현 할 수있다. 

$h_i$는 데이터의 임베딩에 해당하고, $z_i$는 각 데이터에 가한 augmentation에 해당한다. $\tau$는 하이퍼파라미터 temperature값이다.


<!-- ![Alt text](image-3.png) -->

<!-- ![Alt text](image-6.png) -->

$$
\mathcal{L} = \sum_{i=1}^{N} log \exp^ {-\frac{1}{\tau}  sim(h_i,h_i^+)} (Alignment)
$$

$$
+\sum_{i=1}^{N} log \sum_{j=1 }^{N} \exp^{\frac{1}{\tau} sim(h_i,h_j)} (Uniform)
$$


여기서 나오는 sim은 similarity의 약자이고, cosine similarity를 사용하였습니다.

### Notation

i번째 image embedding : $I_i$ 는 row vector

i번째 text embedding : $T_i$

**(단, $I_i,T_j,I_i^+,T_j^+$는 1로 normalize)**

코드 상에서는 cosine similarity를 사용해서 normalize하였습니다.

$$
I_i = \mathbb M(batchsize,d=512)[i]
$$

## Image Text Alignment & Uniform

$$
alignment = -\sum_i tr(II^{+T}+I T^T+ I^+ T^{+T}+TT^{+T})
$$


먼저 위 Object function에서 Uniform식이 아래와 같이 되기 위해서는 convex function라고 가정하고, jensen's inequality를 사용한 결과입니다.

$$
Uniform = \sum_i \sum_{j } I_i \cdot T_j^T + \cdots \\
= sum(II^{+T}+I T^T+ I^+ T^{+T}+TT^{+T})
$$



위 식을 분산과 평균 관점에서 다시 바라보았습니다.
$I_i$가 한개의 임베딩 값이라고 하고, 이 값들은 각 평균과 분산을 갖는다고 하면, 적절한 임베딩은 어느 한 차원으로 쏠리지 않고 적절하게 분산되어서 표현되는것 입니다.

이것에 대한 솔루션으로는 PCA whitening과 batch normalization이가 생각이 납니다. 무엇을 사용해야할지는 알기 위해 수식을 전개해 보았습니다.

$$
I_i = \mu +\sigma_i \\
\mu = \frac{1}{N}\sum_{i \in \chi}^N I_i\\
\therefore \frac{1}{N}\sum_{i \in \chi}^N \sigma_i = 0
$$
$I$만 생각해보면,
$$
\frac{1}{N^2}\sum_{i \in \chi}^N \sum_{j \in \chi}^N I_i \cdot I_j^T 
$$

$$
= \frac{1}{N^2}\sum_{i \in \chi}^N \sum_{j \in \chi }^N (\mu +\sigma_i ) \cdot (\mu +\sigma_j )^T
$$

$$
=\mu \mu^T + \frac{1}{N^2}\sum_{i \in \chi }^N \sum_{j \in \chi}^N \sigma_i \cdot \sigma_j^T \\
=  I \cdot I^T = A
$$

가 되어서 뜻을 해석해보면 임베딩의 평균값을 낮추고, 분산의 곱을 낮추는 식이다. 또한, 위 식은 symmetric matrix이기 때문에 항상 diagonalizable하고, 그 eigen vector는 orthogonal 합니다.

그러한 경우를 eigen decompositoin해서 생각해보자.
$A$라고 놓은 행렬을 $A P_i = \lambda_i P_i$라고 생각해보자. 이때 $P_i$는 $\lambda_i$에 대한 eigen vector입니다.

$$ 
A = P DP^T
$$

이때, $PP^T = E$ 즉, orthogonal하므로, 

$$
A = \sum_i \lambda_i P_i \cdot P_i^T
$$

$\lambda_i$의 어느 한값이 크다는 것은 데이터가 골고루 퍼져있기보단, 한 방향으로 치우쳐져있는것이다. 따라서 위 eigen value값을 골고루 만드느것이 여기서 나온 Uniform의 목적입니다. 

### Flatten Embedding

위 목적을 이루기 위해서 어떻게 해야할까요??

만약에 $I_i$가 normalize 되어있다고 한다면, $tr(A)$의 값은 sum of eigen value이고, constant할것이다. 왜냐하면 diagonal element가 모두 1이기 때문에.
그렇다면 largest eigen value의 값을 줄이고, smallest한 eigen value의 값을 키우면 됩니다. 

만약에 , A의 값들이 모두 양수이고, $sum(P_i \cdot P_i^T)$가 양수라면 sum($A$)를 largest eigen value의 upper bound와 비례한다고 놓을 수 있다. 그래서 위 sum을 줄이는것이, flatten embedding을 하면서 negative pair끼리의 임베딩을 할 수 있습니다. 

SimCSE 논문의 아이디어를 인용하였습니다.
https://arxiv.org/abs/2104.08821

---

## Flatten different Embeddings

하지만 나는 그렇게 조건을 줄 수 없기에, 다른 방식을 생각해야 했습니다. 이유, 다른 임베딩끼리의 표현이기 때문에..

그래서 위처럼 negative pair loss와 Uniform를 하나의 식으로 보지않고, 따로 볼 생각입니다.
이제 I와 T에 대해서 생각해 봅시다.

### negative pair loss

$$
\frac{1}{N^2}\sum_{i \in \chi }^N \sum_{j  \in \Chi}^N I_i \cdot T_j^T 
$$

$$
= \frac{1}{N^2}\sum_{i \in \chi}^N \sum_{j  \in \Chi}^N (\mu^{(Image)} +\sigma_i^{(Image)} ) \cdot (\mu^{(Text)} +\sigma_j^{(Text)} )^T
$$

$$
=\mu^{(Image)} \mu^{(Text)T} + \frac{1}{N^2}\sum_{i \in \chi}^N \sum_{j  \in \Chi}^N \sigma_i \cdot \sigma_j^T
$$

직관적인 의미를 보자면, 이미지와 텍스트의 평균 값을 줄이고, 각 이미지와 텍스트 임베딩의 서로 다른 분산 임베딩을 줄이는 것입니다. 먼저 이걸로, negative pair끼리의 dot product값을 줄여, cosine similarity를 줄일 수 있습니다. 

## Objective function code
```python
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
    def Uniform(self) -> None:
        """Define layers in ther model."""
        raise NotImplementedError
    def Total_loss(self) -> None:
        """Define layers in ther model."""
        raise NotImplementedError
import torch
import torch.nn.functional as F

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
        Alii  = -(self.tau*F.cosine_similarity(self.hi1,self.hi2))
        Alit1 = -(self.tau*F.cosine_similarity(self.hi1,self.ht1))
        Alit2 = -(self.tau*F.cosine_similarity(self.hi2,self.ht2))
        Altt = -(self.tau*F.cosine_similarity(self.ht1,self.ht2))
        return (ALii+ALtt+Alit1+Alit2)#/self.batch_size
    
    def Uniform(self):
        Uniform = torch.empty(batch_size)
        for i in range(batch_size):
            Ii1 = self.hi1[i]
            Ii2 = self.hi2[i]
            Ti1 = self.ht1[i]
            Ti2 = self.ht2[i] 
            Uniformj = 0
            for j in range(batch_size):
                if i!=j:
                    Ij1 = self.hi1[j]
                    Ij2 = self.hi2[j]
                    Tj1 = self.ht1[j]
                    Tj2 = self.ht2[j] 
                    Uniformj+=F.cosine_similarity(Ii1,Ij1,dim=0)
                    Uniformj+=F.cosine_similarity(Ti2,Tj2,dim=0)
                    Uniformj+=F.cosine_similarity(Ii1,Tj1,dim=0)
                    Uniformj+=F.cosine_similarity(Ii2,Tj2,dim=0)
                    # tau 값을 넣으면 너무 커져서 안됨 .. batch size에 따라서 조절해야할듯..        
            Uniform[i] = Uniformj/self.batch_size#*tau
        return Uniform
    
    def Total_loss(self,device):
        alignment  = self.Alignment().to(device)
        Uniform = self.Uniform().to(device)
        return torch.sum(alignment+Uniform)/batch_size
```

# Train
```python
def train(lora_model,device,train_dataloader,train_dataloader2):
    for step,((img,texts),(img2,texts2))in enumerate(zip(train_dataloader,train_dataloader2)):
        optimizer.zero_grad()

        dict1 = {}
        dict1['input_ids'] = texts.to(device)
        dict1['pixel_values'] = img.to(device)

        dict2 = {}
        dict2['input_ids'] = texts2.to(device)
        dict2['pixel_values'] = img2.to(device)
        
        y1 = lora_model(**dict1)
        y2 = lora_model(**dict2)

        image_embeddings = y1.image_embeds
        text_embeddings = y1.text_embeds

        image_embeddings2 = y2.image_embeds
        text_embeddings2 = y2.text_embeds

        # loss function
        loss = SimLoss(hi1=image_embeddings,
                             ht1=text_embeddings,
                             hi2=image_embeddings2,
                             ht2=text_embeddings2)
        alignment = loss.Alignment().to(device)
        Uniform = loss.Uniform().to(device)
        
        total_loss = torch.sum(alignment+Uniform)/batch_size
        total_loss.backward()
        optimizer.step()

        if step%10==0:
            wandb.log({"Learning rate":lr,"total_loss": total_loss.item(), "alignment_loss": alignment_loss,"Uniform_loss":Uniform_loss})
            print(step,"'s batch  ",'  &loss :',round(total_loss.item(),5),'alignment : ',round(alignment.mean().item(),4),' Uniform :',round(Uniform.mean().item(),4))
            print('lr :',lr)
```

## Experiments 

```python
 0%|          | 0/15 [00:00<?, ?it/s]
##################
Epoch :  0
##################
0 's batch     &loss : 0.1842 alignment1,2 :  -0.088 -0.087  Uniform : 0.359
lr : 1.2352941176470589e-05
10 's batch     &loss : 0.1743 alignment1,2 :  -0.088 -0.087  Uniform : 0.349
lr : 3.5882352941176474e-05
20 's batch     &loss : 0.154 alignment1,2 :  -0.082 -0.082  Uniform : 0.319
lr : 4.944045828160822e-05
####################
validation!!
  7%|▋         | 1/15 [02:48<39:23, 168.80s/it]
valid_loss : 0.11913358100822993 valid_alignment -0.07925071428571429 valid_Uniform 0.2775714285714285


									!! 중간 생략 !!


 93%|█████████▎| 14/15 [38:31<02:44, 164.59s/it]
valid_loss : -1.340464472770691 valid_alignment 0.31172999999999995 valid_Uniform -1.964
####################
##################
Epoch :  14
##################
0 's batch     &loss : -1.3409 alignment1,2 :  0.312 0.312  Uniform : -1.965
lr : 2.0051537954535784e-05
10 's batch     &loss : -1.3403 alignment1,2 :  0.311 0.312  Uniform : -1.964
lr : 1.4550932728463223e-05
```

<!-- 
<br>

#### Preprocess & Tokenize
$I_{}^{\prime(i)} = Preprocesser(I^{(i)})$ : CLIP  Preprocess (parameter freeze $\Phi_0$)

$T_{}^{\prime(i)} = Tokenizer(T^{(i)})$ : CLIP Tokenizer (parameter freeze $\Phi_0$)

$Y_{}^{\prime(i)} = Tokenizer(Y^{(i)})$ : CLIP Tokenizer (parameter freeze $\Phi_0$)


<br>

#### Embeddings 
$E_I(I_{}^{\prime(i)};\theta_I)$ : CLIP Image Encoder + LoRA  (learnable params : $\theta_I$)
$E_T(T_{}^{\prime(i)};\theta_T)$ : CLIP Text Encoder + LoRA (learnable params : $\theta_T$)

$E_T(Y_{}^{\prime(i)};\theta_T)$ : CLIP Text Encoder + LoRA (learnable params : $\theta_T$)

#### Concat Image embeddings and Text embeddings

$X^{(i)} = f^{}(E_I(I_{}^{\prime(i)}),E_T(T_{}^{\prime(i)});\Psi)$

$f(\cdot)$ : Image + Text concat models (learnable params : $\Psi$)

<br>

#### Kernel method 

$\psi(X^{(i)})^T \cdot \phi(E_T(Y_{}^{\prime(i)}))$

<br>
miminize KL-divergence
-->







---

### Installation 

#### transformers & peft & loralib

🤗 Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax. Follow the installation instructions below for the deep learning library you are using:

PyTorch installation : [pytorch](https://pytorch.org/get-started/locally/)

TensorFlow 2.0 installation instructions : [tensorflow](https://www.tensorflow.org/install/pip)
Flax installation instructions.[Flax](https://flax.readthedocs.io/en/latest/)


```
python -m venv .env
```
Activate the virtual environment. On Linux and MacOs:

```
source .env/bin/activate
```
Activate Virtual environment on Windows
```
.env/Scripts/activate
```
Now you’re ready to install 🤗 Transformers with the following command:
```
pip install transformers
```
For CPU-support only, you can conveniently install 🤗 Transformers and a deep learning library in one line. For example, install 🤗 Transformers and PyTorch with:
```
/VectorUniverse Project X
├── README.md
├── Data
|  ├── VQA
|  └── Youtube_thumbnails
|       ├── images
|       └── metadata.csv
|
├── node_modules
|  ├── bluebird
|  ├── chalk
|  ├── cli-spinner
|  ├── meow
|  └── object-assign
├── package.json
└── tree.js
```
