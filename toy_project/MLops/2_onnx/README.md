### ONNX download
<br>
```bash
pip install -i https://test.pypi.org/simple/ onnx-weekly
pip install onnxruntime
```

### 실험 환경
맥북 프로(13inch- 2020)

CPU : 1.4GHz quad core Intel Core i7

RAM : 16GB

## ONNX vs Torch

torch inference: 9.868140697479248
ort inference: 1.827899694442749

생각보다 많은 차이가 나므로 inference시에 onnxruntime가 맥북 intel process에서는 유리한것으로 보인다.

또한, 크로스플랫폼을 지원하므로, onnx로 변환한 모델을 다른 언어와 프레임워크 상에서 모델을 서빙할 수 있다.
