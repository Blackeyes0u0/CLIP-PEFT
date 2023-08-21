
import gradio as gr

def image_classifier(inp):
    return {'cat':0.3,'dog':0.7}


import clip,torch
import requests
from PIL import Image
import numpy as np
import torch
from io import BytesIO
import urllib.request

def democlip(url ,text):
    
    if url =='':
        print('SYSTEM : alternative url')
        url = 'https://i.pinimg.com/564x/47/b5/5d/47b55de6f168db65cf46d7d1f0451b64.jpg'
    else:
        print('SYSTEM : URL progressed')
        
    if text =='':
        text ='black desk room girl flower'
    else:
        print('SYSTEM : TEXT progressed')
        
    response = requests.get(url)
    image_bytes = response.content
    
    text = list(text.split(' '))
    print(text)
    """Gets the embedding values for the image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)s
    text_token = clip.tokenize(text).to(device)
    image = preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_token)
        
        logits_per_image, logits_per_text = model(image,text_token)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print('labels probability',probs)
    # return text_features,image_features,probs

    nprobs = []
    for prob in probs[0]:
        nprobs.append(float(prob))
    print(text[0])
    
    mydict = {}
    for i,t in enumerate(text):
        mydict[t] = nprobs[i]
        
    return mydict

# text_emb,img_emb,probs = get_clip_image_description(url)


demo = gr.Interface(
    fn=democlip,
    # inputs = [gr.Image(),gr.Textbox(lable='input prediction')],
    inputs = ['text',gr.Textbox(lable='input prediction')],
    outputs='label'
    # outputs = gr.Textbox(label='Output Box'),
    )
demo.launch()