#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import numpy as np
from PIL import Image


import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


#Denormalize the tensor and convert the tensor to an image for easy visualization
def transform_invert(img_):
    """
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """

#     norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
#     mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
#     std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
#     img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    # Change three chanels from C*H*W to H*W*C
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    # if is a RGB picture
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
        # if is a gray picture
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]) )

    return img_


# In[78]:


#读取图片

BASE_DIR = '/Users/yanghuiran/Graduate-VS/data/'
train_folder = BASE_DIR+'main_train/'
# train_annotation = BASE_DIR+'annotated_train_data/'

files_in_train = sorted(os.listdir(train_folder))
# files_in_annotated = sorted(os.listdir(train_annotation))

# images=[i for i in files_in_train if i in files_in_annotated]
images=[os.path.join(train_folder,i) for i in files_in_train if i.endswith('.jpg')]


# In[79]:


#把图片数据转换成numpy.ndarray
d=len(images)

while d>0:
    print(images[d-1])
    img=Image.open(images[d-1])  #打开图像
    img_array = transform_invert(img)
    print(img_array.shape)
    data.append(img_array)
print(data)


# In[ ]:





# In[7]:


#把图片数据转换成numpy.ndarray
d=len(images)
print(d)


# In[ ]:




