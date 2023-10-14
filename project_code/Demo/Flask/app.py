
import gradio as gr
import clip,torch
import requests
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from io import BytesIO
import urllib.request

# https://hhp-item-resource.s3.ap-northeast-2.amazonaws.com/magazine-resource/magazine/20221017154717/jin._s2.png
# girl bag skirt eye beauty pretty

from selenium import webdriver
from selenium.webdriver.common.by import By

    
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

    
def similarity(v1,v2,type=0):
    if type ==0:
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
    
        return np.dot(v1,v2)/(v1_norm*v2_norm)
    else:
        return np.sqrt(np.sum((v1-v2)**2))

    
def democlip(url ,texts):
    
    if url =='':
        print('SYSTEM : alternative url')
        url = 'https://i.pinimg.com/564x/47/b5/5d/47b55de6f168db65cf46d7d1f0451b64.jpg'
    else:
        print('SYSTEM : URL progressed')
        
    if texts =='':
        texts ='black desk room girl flower'
    else:
        print('SYSTEM : TEXT progressed')
    
    response = requests.get(url)
    image_bytes = response.content
    texts = list(texts.split(' '))

    """Gets the embedding values for the image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)s
    text_token = clip.tokenize(texts).to(device)
    image = preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_token)
        
        logits_per_image, logits_per_text = model(image,text_token)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    word_dict = {'image':{},'text':{}}
    
    ### text 
    for i,text in enumerate(texts):
        word_dict['text'][text] = text_features[i].cpu().numpy()
    
    ### iamge 
    for i,img in enumerate(image):
        word_dict['image'][img] = image_features[i].cpu().numpy()
    
    ###################### PCA of embeddings ########################    
    ## pca of text
    tu,ts,tv = torch.pca_lowrank(text_features,center=True)

    text_pca = torch.matmul(text_features,tv[:,:3])

    ### pca of image
    imgu,imgs,imgv = torch.pca_lowrank(image_features,center=True)

    image_pca = torch.matmul(image_features,imgv[:,:3])

    # return word_dict 
    print(text_pca.shape,image_pca.shape)
    return text_pca,image_pca



def PCA(img_emb, text_emb,n_components = 3):
    x = torch.tensor([[1.,2.,3.,7.],[4.,5.,3.,6.],[7.,9.,8.,9.],[11.,13.,17.,11.]])
    # plz change data type to float or complex 

    print(x.shape)
    u,s,v = torch.pca_lowrank(x,q=None, center=False,niter=2)

    u.shape,s.shape,v.shape

    u@torch.diag(s)@v.T

    # torch.matmul(x,v[:,:3])
    pass



# NODE type

# PCA type.

# ELSE type. 
demo = gr.Interface(
    fn=democlip,
    # inputs = [gr.Image(),gr.Textbox(lable='input prediction')],
    inputs = ['text',gr.Textbox(lable='input prediction')],
    # outputs='label'
    outputs = [gr.Textbox(label='text pca Box'),gr.Textbox(label='image pca Box')]
    )
demo.launch()