
# coding: utf-8

# In[14]:


import io
import requests
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from utils.timer import Timer


# In[39]:


dtype = torch.cuda.FloatTensor
resnet101=models.resnet101(pretrained=True) 
resnet101.cuda()
resnet101.eval()


# In[40]:


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])


# In[41]:


#response = requests.get('cat.jpg')
img_pil = Image.open('/home/nvidia/Desktop/data/mosaik5.jpg')
#img_pil


# In[42]:


img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)


# In[43]:


img_variable = Variable(img_tensor)
timer = Timer()
timer.tic()
out = resnet101(img_variable.cuda())
timer.toc()


# In[28]:


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}


# In[12]:


labels


# In[37]:


#print(labels[out.data.numpy().argmax()])


# In[38]:


print("Time for Image classification {:.3f}s ".format(timer.total_time()))

