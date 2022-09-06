#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logNoAugmentation = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/AugmentationTest/log_NoAugmentation.csv')
logAugmentation = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/AugmentationTest/log_Augmentation.csv')
#log75 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/75%MaskedRate/log75.csv')
plt.figure(figsize=[10, 10])
#plt.plot(logNoAugmentation.epoch, logNoAugmentation.train_loss,'b.-',label="No Audio Augmentation",color = "red")
plt.plot(logAugmentation.epoch, logAugmentation.train_loss,'b.-',label="50% Masked",color = "green")
plt.plot(log75.epoch, log75.train_loss,label='75% Masked',color = "red")

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend(loc='best')
plt.show()


# In[14]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log0 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/large/40%/log_new.csv')
log1 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/large/AugmentationTest/log_NoAugmentation.csv')
log2 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/large/60%/log60.csv')
log3 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/large/70%/log70.csv')
log4 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/large/80%/log80.csv')


#log75 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/75%MaskedRate/log75.csv')
plt.figure(figsize=[10, 10])
#plt.plot(logNoAugmentation.epoch, logNoAugmentation.train_loss,'b.-',label="No Audio Augmentation",color = "red")
plt.plot(log0.epoch, log0.train_loss,'b.-',label="40% Masked",color = "black")
plt.plot(log1.epoch, log1.train_loss,'b.-',label="50% Masked",color = "green")
plt.plot(log2.epoch, log2.train_loss,label='60% Masked',color = "red")
plt.plot(log3.epoch, log3.train_loss,label='70% Masked',color = "blue")
plt.plot(log4.epoch, log4.train_loss,label='80% Masked',color = "yellow")


plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend(loc='best')
plt.show()


# In[19]:


#图形的数据增强方式对语音自监督的影响

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log2_1 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/base/50%hasVisionAugment/log_new.csv')
log2_2 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/base/50%noVisionAugment/log_new.csv')

#log75 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/75%MaskedRate/log75.csv')
plt.figure(figsize=[10, 12])
#plt.plot(logNoAugmentation.epoch, logNoAugmentation.train_loss,'b.-',label="No Audio Augmentation",color = "red")
plt.plot(log2_1.epoch, log2_1.train_loss,'b.-',label="Including Vision Enhancement",color = "red")
plt.plot(log2_2.epoch, log2_2.train_loss,label='Without Vision Enhancement',color = "black")


plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend(loc='best')
plt.show()


# In[23]:


#模型深度对语音自监督的影响

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# base
log3_1 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/base/50%hasVisionAugment/log_new.csv')
# large
log3_2 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/large/AugmentationTest/log_NoAugmentation.csv')

#log75 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/75%MaskedRate/log75.csv')
plt.figure(figsize=[8, 5])
#plt.plot(logNoAugmentation.epoch, logNoAugmentation.train_loss,'b.-',label="No Audio Augmentation",color = "red")
plt.plot(log3_1.epoch, log3_1.train_loss,'b.-',label="ViT-Base",color = "red")
plt.plot(log3_2.epoch, log3_2.train_loss,label='ViT-Large',color = "black")


plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend(loc='best')
plt.show()


# In[21]:


#模型深度对语音自监督的影响

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 80epochs
log4_1 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/base/80epoches50%/log_new.csv')
# 50epochs
log4_2 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/base/50%hasVisionAugment/log_new.csv')

#log75 = pd.read_csv('/Users/yanghuiran/Graduate-VS/output_dir/75%MaskedRate/log75.csv')
plt.figure(figsize=[8, 5])
#plt.plot(logNoAugmentation.epoch, logNoAugmentation.train_loss,'b.-',label="No Audio Augmentation",color = "red")
plt.plot(log4_1.epoch, log4_1.train_loss,'b.-',label='80 epochs',color = "red")
plt.plot(log4_2.epoch, log4_2.train_loss,label='50 epochs',color = "black")


plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend(loc='best')
plt.show()

