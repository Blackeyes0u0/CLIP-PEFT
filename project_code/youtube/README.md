## purpose

error < purpose error - 어필 할만 한 오류 추정 -clip을 통한 성능 비교.

metric : Precision,recall, accuracy 결정

### end 2 end pipline

성과측정법

### 병목현상 ;logging


---

### 개선

new data, hyperparams tuning(grid, random search)

+pretrained 결정
+optimizer 결정
  normalization & regularization

+ early stopping

### Define Datasets
$$
x^{(i)},y^{(i)} \to    \tilde x^{(i)}, \tilde y^{(i)} 
$$

### Define models
$$
\hat y = f_{\theta}(x;\theta)
$$

##### Define Loss function & objective functions
$$
L_{\phi}(\hat y,y^{(i)};\phi) : non-convex
$$

### Define optimizer 
ex) Adam
$$
\theta \leftarrow \theta - \epsilon \frac{\hat s}{\sqrt{\hat r}+ \delta}
$$

#Purpose :
$$
f_{\theta}^{*}(x^{(i)}) \simeq y^{(i)}
$$