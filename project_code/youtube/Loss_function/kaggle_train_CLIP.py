from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts , CyclicLR, ExponentialLR,StepLR, CosineAnnealingLR
import torch.optim as optim
from tqdm import tqdm

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# loss_alignment = nn.MSELoss()
loss_alignment = nn.CrossEntropyLoss()

# cyclic LR
# optimizer = optim.Adam(lora_model.parameters(), lr=2e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
# scheduler = CyclicLR(optimizer, base_lr=5e-5, max_lr=0.003, step_size_up=5, step_size_down=5, mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.95, last_epoch=-1)
## 만약에 cosineannealingwarmuprestart쓰는경우
optimizer = optim.Adam(lora_model.parameters(), lr = 5e-6)#,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=70, T_mult=1, eta_max=2e-5,  T_up=17, gamma=0.9)

# loss & learing rate 변화 관측
logging = {}
logging['lr_values'] = []
logging['loss_values'] = []
logging['alignment_loss_values'] = []
logging['anisotropy_loss_values'] = []
logging['valid_loss_values'] = []


EPOCH = 15
tau = 1/0.05
# add your own code to track the training progress.

lora_model.to(device)
lora_model.train()
for epoch in tqdm(range(EPOCH)):
    print('##################')
    print("Epoch : ",epoch)
    print('##################')
    
    for step,((img,texts),(img2,texts2))in enumerate(zip(train_dataloader,train_dataloader2)):
        optimizer.zero_grad()

        images= img.to(device)
        texts = texts.to(device)
        
        images2= img2.to(device)
        texts2 = texts2.to(device)
        
        dict1 = {}
        dict1['input_ids'] = texts
        dict1['pixel_values'] = images
        
        dict2 = {}
        dict2['input_ids'] = texts2
        dict2['pixel_values'] = images2

        y1 = lora_model(**dict1)
        y2 = lora_model(**dict2)
        
        image_embeddings = y1.image_embeds
        text_embeddings = y1.text_embeds
        
        image_embeddings2 = y2.image_embeds
        text_embeddings2 = y2.text_embeds
        
        # cosine
        # alignment_ii = -torch.log(F.cosine_similarity(image_embeddings,image_embeddings2))
        h1 = torch.randn(batch_size,512)
        alignment = -torch.log(F.cosine_similarity(h1,h1))
        alignment_it1 = -(tau*F.cosine_similarity(image_embeddings,text_embeddings))
        alignment_it2 = -(tau*F.cosine_similarity(image_embeddings2,text_embeddings2))
        
        # batch_size로 나눠주기. normalize는 어떻게 해주지?
        alignment_it1 = alignment_it1/batch_size
        alignment_it2 = alignment_it2/batch_size
        
        alignment_it1.to(device)
        alignment_it2.to(device)
        anisotropy = torch.empty_like(alignment_it1)
        
        for i in range(batch_size):
            I1 = image_embeddings[i]
            I2 = image_embeddings2[i]
            T1 = text_embeddings[i]
            T2 = text_embeddings2[i]
            anisotropyj = 0
            for j in range(batch_size): 
                if i!=j:
                    anisotropyj+= F.cosine_similarity(image_embeddings2[i,:],text_embeddings2[j,:],dim=0)
                    anisotropyj+= F.cosine_similarity(image_embeddings[i,:],text_embeddings[j,:],dim=0)
            # tau 값을 넣으면 너무 커져서 안됨 .. batch size에 따라서 조절해야할듯..        
            anisotropy[i] = anisotropyj/batch_size#*tau
        anisotropy.to(device)            
        
        # loss function
        total_loss = torch.sum(alignment_it1+alignment_it2+anisotropy)/batch_size
        total_loss.backward()
        optimizer.step()
        
        scheduler.step()
        
        ## logging
        lr = optimizer.param_groups[0]['lr']
        logging['lr_values'].append(lr)
        logging['loss_values'].append(round(total_loss.item(),4))

        alignment_loss = round(alignment_it1.mean().item(),3)+round(alignment_it2.mean().item(),3)
        anisotropy_loss = round(anisotropy.mean().item(),3)
        
        logging['alignment_loss_values'].append(alignment_loss)
        logging['anisotropy_loss_values'].append(anisotropy_loss)

        if step%10==0:
            print(step,"'s batch  ",'  &loss :',round(total_loss.item(),4),'alignment1,2 : ',round(alignment_it1.mean().item(),3),round(alignment_it2.mean().item(),3),' anisotropy :',round(anisotropy.mean().item(),3))
            print('lr :',lr)
        
        
    validation_test(lora_model,valid_dataloader,valid_dataloader2)
#     logging['valid_loss_values'].append()
       