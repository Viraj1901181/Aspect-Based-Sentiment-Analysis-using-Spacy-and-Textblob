#!/usr/bin/env python
# coding: utf-8

# ## Estimate sentiment for specific topics or attributes

# In[1]:


import pyforest


# In[2]:


dir(pyforest)


# ## One of the most common goals with NLP is to analyze text and extract insights. You can find countless tutorials on how to perform sentiment analysis, but the typical way that’s used is not always enough.
# 
# ### When you pass a sentence like this.
# 
# We had some amazing food yesterday. But the next day was very boring.
# 
# #### A sentence-by-sentence sentiment analysis algorithm would produce something like this.

# ## We had some amazing food yesterday. POSITIVE
# 
# ## But the next day was very boring. NEGATIVE

# #### But there are times when you want your sentiment analysis to be aspect-based, or otherwise called topic-based.
# 
# In this article, we will build a very simplistic aspect-based sentiment analysis that’s able to pick up generic concepts and understand the sentiments around them. In our previous example, that would mean something like:
# 

# ### Amazing food POSITIVE
# 
# ### Very boring day NEGATIVE

# The aspects, or topics, in this case, are food and day. By performing sentiment analysis based on aspects we analyze large pieces of text and extract insights.
# 
# For example, if you monitor customer reviews, or call transcripts, you can look for aspects that have some sentiment attached and extract insights on how to improve.
# 
# For this article, we will be using spacy, a natural language processing library in Python along with Textblob which offers simple tools for sentiment analysis and text processing.

# In[4]:


# We get started by importing spacy
import spacy
nlp = spacy.load("en_core_web_sm")


# Let’s also define a few simple test sentences.

# In[5]:


sentences = [
  'The food we had yesterday was delicious',
  'My time in Italy was very enjoyable',
  'I found the meal to be tasty',
  'The internet was slow.',
  'Our experience was suboptimal'
]


# Our first goal is to split our sentences in a way so that we have the target aspects (e.g. food) and their sentiment descriptions (e.g. delicious).

# In[6]:


for sentence in sentences:
  doc = nlp(sentence)
  for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
      token.pos_,[child for child in token.children])


# For each token inside our sentences, we can see the dependency thanks to spacy’s dependency parsing and the POS (Part-Of-Speech) tags. We’re also paying attention to the child tokens, so that we’re able to pick up intensifiers such as “very”, “quite”, and more.
# 
# Disclaimer: Our current simplistic algorithm may not be able to pick up semantically important information such as the “not” in “not great” at the moment. That would be crucial to account for in a real-life application.
# 
# Let’s see how to pick up the sentiment descriptions first.

# In[8]:


for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  for token in doc:
    if token.pos_ == 'ADJ':
      descriptive_term = token
  print(sentence)
  print(descriptive_term)


# You can see that our simplistic algorithm picks up all the descriptive adjectives such as delicious, enjoyable, and tasty. But what’s currently missing are intensifiers, like “very” for example.
# f

# In[9]:


for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  for token in doc:
    if token.pos_ == 'ADJ':
      prepend = ''
      for child in token.children:
        if child.pos_ != 'ADV':
          continue
        prepend += child.text + ' '
      descriptive_term = prepend + token.text
  print(sentence)
  print(descriptive_term)


# As you can see, this time around we picked up very enjoyable as well. Our simplistic algorithm is able to pick up adverbs. It checks for child tokens for each adjective and picks up the adverbs such as “very”, “quite”, etc.
# 
# In a regular scenario, we would need to catch negations such as “not” as well, but this is outside the scope of this article. But you are encouraged to practice and make it more advanced afterward if you’d like.
# 
# We’re now ready to identify the targets that are being described.

# In[11]:


aspects = []
for sentence in sentences:
  doc = nlp(sentence)
  descriptive_term = ''
  target = ''
  for token in doc:
    if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
      target = token.text
    if token.pos_ == 'ADJ':
      prepend = ''
      for child in token.children:
        if child.pos_ != 'ADV':
          continue
        prepend += child.text + ' '
      descriptive_term = prepend + token.text
  aspects.append({'aspect': target,
    'description': descriptive_term})
print(aspects)


# Now our solution is starting to look more complete. We’re able to pick up aspects, even though our application doesn’t “know” anything beforehand. We haven’t hardcoded the aspects such as “food”, “time”, or “meal”. And we also haven’t hardcoded the adjectives such as “tasty”, “slow”, or “enjoyable”.
# 
# Our application picks them up based on the simple rules that we set.
# 
# There are times when you may want to find the topics first and then identify them in your text while ignoring topics or aspects that are not that common.
# 
# To do that, you would need to work on Topic Modeling before moving on to the sentiment analysis part of the solution. There’s a great guide on Towards Data Science that explains the Latent Dirichlet Allocation which you can use for Topic Model

# Now that we successfully extracted the aspects and descriptions, it’s time to classify them as positive or negative. 
# 
# The goal here is to help the computer understand that tasty food is positive, while slow internet is negative. Computers don’t understand English, so we will need to try a few things before we have a working solution.
# 
# We will start off by using the default TextBlob sentiment analysis.

# In[12]:


from textblob import TextBlob
for aspect in aspects:
  aspect['sentiment'] = TextBlob(aspect['description']).sentiment
print(aspects)


# TextBlob is a library that offers sentiment analysis out of the box. It has a bag-of-words approach, meaning that it has a list of words such as “good”, “bad”, and “great” that have a sentiment score attached to them. It is also able to pick up modifiers (such as “not”) and intensifiers (such as “very”) that affect the sentiment score.
# 
# If we look at our results, I can say it’s definitely not looking bad! I would agree with all of them, but the only problem is that tasty and suboptimal are considered neutral. It seems that they’re not part of TextBlob’s dictionary and as such, they are not picked up.
# 
# Another potential issue is that some descriptive terms or adjectives can be positive in some cases and negative in others, depending on the word they’re describing. The default algorithm used by TextBlob is not able to know that cold weather can be neutral, cold food can be negative while a cold drink can be positive.
# 
# The good thing is that TextBlob allows you to train a NaiveBayesClassifier using a very simple syntax that’s easy for anyone to understand, which we will use to improve our sentiment analysis.
# 
# To be able to use it though, you will need to execute the following to download the required corpora: python -m textblob.download_corpora
# 

# In[13]:


from textblob.classifiers import NaiveBayesClassifier
# We train the NaivesBayesClassifier
train = [
  ('Slow internet.', 'negative'),
  ('Delicious food', 'positive'),
  ('Suboptimal experience', 'negative'),
  ('Very enjoyable time', 'positive'),
  ('delicious food.', 'neg')
]
cl = NaiveBayesClassifier(train)


# In[14]:


# And then we try to classify some sample sentences.
blob = TextBlob("Delicious food. Very Slow internet. Suboptimal experience. Enjoyable food.", classifier=cl)


# In[15]:


for s in blob.sentences:
  print(s)
  print(s.classify())


# As you can see, the sentences we passed are not exactly what we used in the training examples, but it’s still able to correctly predict the sentiments of all the phrases.

# ## Conclusion, Scope and Further Work Possible

# You can use this solution as a starting point for a more complex aspect-based sentiment analysis solution. To do that you would need to improve the dependency parsing process to extract more accurate information, but also more types of data. A great way to do that is using spacy’s DependencyMatcher which allows you to match patterns using custom rules.
# 
# As for the sentiment analysis part, ideally, you want to label a lot of data so that you can create more advanced classifiers with a higher amount of accuracy. There are tons of binary (or categorical when you include neutral in the mix) classifications that you can perform using keras, TensorFlow, or other machine learning libraries and tools.
# 
# If you have pre-labeled data that’s very helpful. If you don’t, you can create an initial analysis using a simple tool like TextBlob, and then instead of deciding on the sentiment for each phrase, you can choose if you agree with TextBlob or not which is a lot quicker than deciding the sentiment from scratch.
