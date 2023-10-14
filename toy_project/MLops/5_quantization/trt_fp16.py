import time
import argparse
import onnxruntime as ort
from transformers import BertTokenizerFast

cuda_providers = [
    "CUDAExecutionProvider"
]
tensorrt_providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 1,
    }),
]
tensorrt_fp16_providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 1,
        'trt_fp16_enable': True,
    }),
]

providers = {"CUDA": cuda_providers, "TensorRT": tensorrt_providers, "TensorRT-fp16": tensorrt_fp16_providers}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def benchmark(key):
    sess = ort.InferenceSession('model/model.onnx', providers=providers[key])

    # first inference for initializing gpu
    encoded_input = tokenizer(" ".join(["hello"]*510), return_tensors='np')
    output = sess.run(None, input_feed=dict(encoded_input))

    start_time = time.time()
    for _ in range(1000):
        output = sess.run(None, input_feed=dict(encoded_input))
    end_time = time.time()
    return "it takes {} seconds with {} provider".format(end_time - start_time, key)


print("\n".join([benchmark(key) for key in providers.keys()]))
