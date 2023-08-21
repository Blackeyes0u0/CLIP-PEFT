import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

import gradio as gr
import clip,torch
import requests
from PIL import Image
import numpy as np
import torch
from io import BytesIO
import urllib.request

from selenium import webdriver
from selenium.webdriver.common.by import By

def similarity(v1,v2,type=0):
    if type ==0:
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
    
        return torch.dot(v1,v2)/(v1_norm*v2_norm)
    if type ==1:
        return torch.sqrt(torch.sum((v1-v2)**2))
    
def test2():
    driver = webdriver.Chrome()        #웹드라이버가 있는 경로에서 Chrome을 가져와 실행-> driver변수
    
    driver.get('https://www.hiphoper.com/')     #driver변수를 이용해 원하는 url 접속
    
    imgs = driver.find_elements(By.CSS_SELECTOR,'img.card__image')       #css selector를 이용해서 'tag이름.class명'의 순으로 인자를 전달
    result = []     #웹 태그에서 attribute 중 src만 담을 리스트
    
    for img in imgs:                                #모든 이미지들을 탐색
        # print(img.get_attribute('src'))         #이미지 주소를 print
        result.append(img.get_attribute('src'))     #이미지 src만 모아서 리스트에 저장
    
    driver.quit()
    
    return result



# initializing
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def preprocess_image(image_bytes) -> torch.tensor:
    _, preprocess = clip.load("ViT-B/32")
    return preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0)

def get_prediction():
    # result = test2()
    # urls = result[:7]
    urls = ['https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221017154717/jin._s2.png',
    'https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221017154558/jackystar0110_309143377_133138599466977_6751708104386392316_n.png',
    'https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221017154457/95k__deliver.png',
    'https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221017154331/_xjunk.png',
    'https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221013161531/rickuns.png',
    'https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221013161434/265mm_xxl%2820.png',
    'https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221013161349/2x_xs2%282%29.png']
    
    text = 'girl boy band dog cat room'
    text = list(text.split(' '))

    ##################### IMAGE #####################
    # inital images.
    
    image_features = torch.tensor([])
    for url in urls:
        
        response = requests.get(url)
        image_bytes = response.content #
        
        # type of iamge preprocess
        image = transform_image(image_bytes=image_bytes)
        image = preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(image)
        image_features = torch.concat([image_features,image_feature],dim=0) #torch.Size([-1,512])

    ##################### TEXT #####################
    text_token = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        # logits_per_image, logits_per_text = model(image,text_token)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
    result1 = []
    for i,vec in enumerate(text_features):
        result1.append(similarity(image_features[0],vec,type=1))
        # result1.append(similarity(text_features[0],vec,type=1))


    ##################### P C A #####################
    ## pca of text
    tu,ts,tv = torch.pca_lowrank(text_features,center=True)
    text_pca = torch.matmul(text_features,tv[:,:3])

    ### pca of image
    imgu,imgs,imgv = torch.pca_lowrank(image_features,center=True)
    image_pca = torch.matmul(image_features,imgv[:,:3]).numpy()

    # return text_pca,image_pca
    vec_universe = {}
    vec_universe['image'] = {urls[i]:ipca for i,ipca in enumerate(image_pca)}
    vec_universe['text'] = {text[i]:tpca for i,tpca in enumerate(text_pca)}
    return vec_universe


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # file = request.files['file']
        # img_bytes = file.read()
        vec_universe = get_prediction()
        return jsonify(vec_universe)

if __name__ == '__main__':
    app.run()