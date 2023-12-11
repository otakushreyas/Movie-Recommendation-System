#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Import the datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# # Data Preprocessing

# In[5]:


#Merge the dataframes on the basis of movie Titles and name the new dataframe as 'movies'.
movies = movies.merge(credits, on='title')


# In[6]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[7]:


#Remove null values
movies.isnull().sum()


# In[8]:


movies.dropna(inplace=True)


# In[9]:


#Remove duplicate values
movies.duplicated().sum()


# In[10]:


movies.iloc[0].genres


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# Convert it to
# {['Action','Adventure','Fantasy','Scify']}

# In[11]:


import ast

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert2(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if(counter!=3):
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

def convert3(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[12]:


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert2)     #Get only top 3 actors
movies['crew'] = movies['crew'].apply(convert3)     #Get only the director




movies['overview'] = movies['overview'].apply(lambda x:x.split())



# In[15]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])   
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])     




movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']



# In[18]:


new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[19]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# # Stemming

# In[20]:


#if some similar words like 'loved,loving,love' etc are getting repeated then use stemming
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[21]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))     #ps.stem() will convert all the forms of words to its V1 form. Like loved, loving will b converted to love.
    return " ".join(y)


# In[22]:


new_df['tags']=new_df['tags'].apply(stem)


# In[26]:


new_df


# # Vectorization

# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[24]:


#Convert top 5000 words to vectors
vectors=cv.fit_transform(new_df['tags']).toarray()
vectors


# In[25]:


cv.get_feature_names()


# # Finding Similarity

# In[27]:


from sklearn.metrics.pairwise import cosine_similarity


# In[28]:


similarity = cosine_similarity(vectors)




sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]


# In[34]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[35]:

#print("Please enter your recently watched movie")

recommend('Avatar')


# In[ ]:




