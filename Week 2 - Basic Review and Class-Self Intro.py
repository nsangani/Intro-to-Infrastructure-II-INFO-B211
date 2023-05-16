#!/usr/bin/env python
# coding: utf-8

# In[1]:


print(2**3+2*2)
print(2**(3+2*2))
print(2**(3+2)*2)


# In[7]:


x = 0.2-0.1 #approximation are not same
y=0.3-0.2
a = 2-1
b = 3-2
if x == y:
    print('true')
else:
    print('false')
a == b


# In[15]:


def Message(name, message, signature = "Neel Sangani"):
    print('Hello', name + '!' + '\n')
    print(message)
    print(signature)
    
Message('Marshmello', 'Welcome to the page.')


# In[20]:


my_list = [0,1,2]
my_list.append(3)
print(my_list)
my_list.reverse()
print(my_list)
my_list.index(3)


# In[ ]:


class Animal():
    def __init__(self, species,sex,height,weight,personality,habitat,food__Type,food__Amount, status):
        self.species = species
        self.sex = sex
        self.height = height
        self.weight = weight
        self.personality = personality
        self.habitat = habitat
        self.food__Type = food__Type
        self.food_amount = food__Amount
        self.status = status
        

