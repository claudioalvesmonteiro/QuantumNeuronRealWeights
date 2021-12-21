#!/usr/bin/env python
# coding: utf-8

# In[8]:


from hsgs import dec_to_bin
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


# In[9]:


def makeAllBin():
    list = []
    
    for i in range(256):
        bin1 = dec_to_bin(i, 8)
        inputV = [int(bin1[0]), int(bin1[1]), int(bin1[2]), int(bin1[3])]
        
        for j in range(256):
            bin2 = dec_to_bin(j, 4)
            weightV = [int(bin2[0]), int(bin2[1]), int(bin2[2]), int(bin2[3])]
            list.append((inputV, weightV))
            
    return list

    


# In[10]:


makeAllBin()


# In[11]:


# text_file = open("Output.txt", "w")
# text_file.write("Purchase Amount: %s" % TotalAmount)
# text_file.close()

