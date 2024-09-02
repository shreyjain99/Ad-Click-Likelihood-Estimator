#!/usr/bin/env python
# coding: utf-8

# In[1]:


# A Quick recap of the problem...

# Given a (Uid, Qid, Adid) under certain Ad setting (such as pos, depth etc.) we want to predict the Ad CTR.

# Recall, CTR(Ad) = #clicks/#impressions


# In[ ]:


TASK 2 DESCRIPTION

Search advertising has been one of the major revenue sources of the Internet industry for years. A key technology behind search advertising is to predict the click-through rate (pCTR) of ads, as the economic model behind search advertising requires pCTR values to rank ads and to price clicks. In this task, given the training instances derived from session logs of the Tencent proprietary search engine, soso.com, participants are expected to accurately predict the pCTR of ads in the testing instances.

TRAINING DATA FILE   

The training data file is a text file, where each line is a training instance derived from search session log messages. To understand the training data, let us begin with a description of search sessions.   

A search session refers to an interaction between a user and the search engine. It contains the following ingredients: the user, the query issued by the user, some ads returned by the search engine and thus impressed (displayed) to the user, and zero or more ads that were clicked by the user. For clarity, we introduce a terminology here. The number of ads impressed in a session is known as the ’depth’. The order of an ad in the impression list is known as the ‘position’ of that ad. An Ad, when impressed, would be displayed as a short text known as ’title’, followed by a slightly longer text known as the ’description’, and a URL (usually shortened to save screen space) known as ’display URL’.   

We divide each session into multiple instances, where each instance describes an impressed ad under a certain setting  (i.e., with certain depth and position values).  We aggregate instances with the same user id, ad id, query, and setting in order to reduce the dataset size. Therefore, schematically, each instance contains at least the following information:

UserID 
AdID 
Query 
Depth 
Position 
Impression 
the number of search sessions in which the ad (AdID) was impressed by the user (UserID) who issued the query (Query).

Click 
the number of times, among the above impressions, the user (UserID) clicked the ad (AdID).   

Moreover, the training, validation and testing data contain more information than the above list, because each ad and each user have some additional properties. We include some of these properties into the training, validation  and the testing instances, and put other properties in separate data files that can be indexed using ids in the instances. For more information about these data files, please refer to the section ADDITIONAL DATA FILES. 

Finally, after including additional features, each training instance is a line consisting of fields delimited by the TAB character: 

1. Click: as described in the above list. 

2. Impression: as described in the above list. 

3. DisplayURL: a property of the ad. 

The URL is shown together with the title and description of an ad. It is usually the shortened landing page URL of the ad, but not always. In the data file,  this URL is hashed for anonymity. 

4. AdID: as described in the above list. 

5. AdvertiserID: a property of the ad. 

Some advertisers consistently optimize their ads, so the title and description of their ads are more attractive than those of others’ ads. 

6. Depth: a property of the session, as described above.   

7. Position: a property of an ad in a session, as described above. 

8. QueryID:  id of the query. 

This id is a zero‐based integer value. It is the key of the data file 'queryid_tokensid.txt'.

9. KeywordID: a property of ads. 

This is the key of  'purchasedkeyword_tokensid.txt'. 

10. TitleID: a property of ads. 

This is the key of 'titleid_tokensid.txt'. 

11. DescriptionID: a property of ads. 

 This is the key of 'descriptionid_tokensid.txt'. 

12. UserID 

This is the key of 'userid_profile.txt'.  When we cannot identify the user, this field has a special value of 0.

 ADDITIONAL DATA FILES

There are five additional data files, as mentioned in the above section: 

1. queryid_tokensid.txt 

2. purchasedkeywordid_tokensid.txt 

3. titleid_tokensid.txt 

4. descriptionid_tokensid.txt 

5. userid_profile.txt 

Each line of the first four files maps an id to a list of tokens, corresponding to the query, keyword, ad title, and ad description, respectively. In each line, a TAB character separates the id and the token set.  A token can basically be a word in a natural language. For anonymity, each token is represented by its hash value.  Tokens are delimited by the character ‘|’. 

Each line of ‘userid_profile.txt’ is composed of UserID, Gender, and Age, delimited by the TAB character. Note that not every UserID in the training and the testing set will be present in ‘userid_profile.txt’. Each field is described below: 

1. Gender: 

'1'  for male, '2' for female,  and '0'  for unknown. 

2. Age: 

'1'  for (0, 12],  '2' for (12, 18], '3' for (18, 24], '4'  for  (24, 30], '5' for (30,  40], and '6' for greater than 40. 


# In[2]:


# Quick reminder, there are total of 7 files involved in this problem listed as follows:

# 1. training file

# 2. testing file + solution file

# 3. user file: corr. to every user we maintain their gender & age info.

# 4. Ad title file: corr. to every ad we maintain their titles(hashed). We have used count of words in title.

# 5. Ad description file: corr. to every ad we maintain its desc(hashed).We have used count of words in Ad desc.

# 6. user query file: corr. to every qid we have the query(issued). We have used count of words used in a query.

# 7. keyword file: We have used count of words in keyword.


# # Source/Useful Links:  
# 
# Source : https://www.kaggle.com/c/kddcup2012-track2 <br>
# 
# pdf : https://jyunyu.csie.org/docs/pubs/kddcup2012paper.pdf 
# 
# 
# 

# In[3]:


# Part 1 - Data Preparation...

# Goal of Part 1: Make training & testing data ready for building model.


# In[4]:


# Loading Libraries...

import pandas as pd
import numpy as np


# In[5]:


#  The training set contains 155,750,158 instances but we are limiting ourselves to 5,000,000


# In[6]:


# 1.1.1. Loading training data...

column  = ['clicks', 'impressions', 'AdURL', 'AdId', 'AdvId', 'Depth', 'Pos', 'QId', 'KeyId', 'TitleId', 'DescId', 'UId']
train   = pd.read_csv('track2/track2/training.txt', sep='\t', header=None, names=column,nrows = 5000000)
train.head()


# In[7]:


# we observe that some categories come with only a few or even no instances.

# Computing the click-through rate directly for those categories would result in inaccurate estimations 

# because of the insuﬃcient statistics. Thus, we apply smoothing methods during click-through rate estima-tion. 

# We mainly use a simple additive smoothing  pseudo-CTR = click + α × β #impression + β

# and we name it pseudo click-through rate (pseudo-CTR). In our experiments, we set α as 0.05 and β as 75. 


# In[8]:


# Add target variable CTR as #clicks / #impression

train['CTR'] = train['clicks'] * 1.0 / train['impressions']

#adding relative position as a new feature
train['RPosition'] = train['Depth'] - train['Pos'] * 1.0 / train['Depth']

# Add predicted CTR as #clicks + ab / #impressions + b
train['pCTR'] = (1.0 * train['clicks'] + 0.05 * 75) / (train['impressions'] + 75)

train.head()


# In[9]:


train.shape


# In[10]:


# The test set contains 20,297,594 instances, we are limiting ourselves to 1,000,000


# In[11]:


# 1.1.2 Loading test data...

column  = ['AdURL', 'AdId', 'AdvId', 'Depth', 'Pos', 'QId', 'KeyId', 'TitleId', 'DescId', 'UId']
test = pd.read_csv('test/test.txt', sep='\t', header=None, names=column, nrows = 1000000)

test.head()


# In[12]:


test.shape


# In[13]:


# As you can see the test data provided for this problem doesn't provide the means to calculate target variable CTR.

# However at the end of competition Kaggle posted a solution.txt file that provides the #Clicks & #Impressions corr.

# to each testing instance. Hence, I have merged the solution.txt with test dataset.


# In[14]:


# Loading test data solution...

solution = pd.read_csv('KDD_Track2_solution.csv',  nrows=1000000)
solution.rename(columns = {'I clicks':'clicks'}, inplace = True)
solution = solution[['clicks', 'impressions']].copy()
solution.head()


# In[15]:


solution.shape


# In[16]:


# concatanating test & solution 

test = pd.concat([solution, test], axis=1)
test.head()


# In[17]:


# Add target variable CTR to test set...

test['CTR'] = test['clicks'] * 1.0 / test['impressions']
test['RPosition'] = test['Depth'] - test['Pos'] * 1.0 / test['Depth']

# Add predicted CTR as #clicks + ab / #impressions + b
test['pCTR'] = (1.0 * test['clicks'] + 0.05 * 75) / (test['impressions'] + 75)

test.head()


# In[18]:


test.shape


# In[19]:


#  Now, we will load additional files provided in the problem, extract useful info. from them & merge
#  with training & testing datasets...


# In[20]:


def count(sentence):
    '''
        (str) -> (int)
        Returns no. of words in a sentence.
    '''
    return len(str(sentence).split('|'))


# In[21]:


# Load User Data..

user_col  = ['UId', 'Gender', 'Age']
user      = pd.read_csv('track2/track2/userid_profile.txt', sep='\t', header=None, names=user_col)

# Load Query Data..

query_col = ['QId', 'Query']
query     = pd.read_csv('track2/track2/queryid_tokensid.txt', sep='\t', header=None, names=query_col)

# Load Ad Description Data..

desc_col  = ['DescId', 'Description']
desc      = pd.read_csv('track2/track2/descriptionid_tokensid.txt', sep='\t', header=None, names=desc_col)

# Load Ad Title Data..

title_col = ['TitleId', 'Title']
title     = pd.read_csv('track2/track2/titleid_tokensid.txt', sep='\t', header=None, names=title_col)

# Load Keyword Data..

key_col  = ['KeyId', 'Keyword']
keyword  = pd.read_csv('track2/track2/purchasedkeywordid_tokensid.txt', sep='\t', header=None, names=key_col)

# Count no. of tokens in a query issued by a user.

query['QCount'] = query['Query'].apply(count)
del query['Query']

# Count no. of tokens in title of an advertisement.

title['TCount'] = title['Title'].apply(count)
del title['Title']

# Count no. of tokens in description of an advertisement.

desc['DCount'] = desc['Description'].apply(count)
del desc['Description']

# Count no. of tokens in purchased keyword.

keyword['KCount'] = keyword['Keyword'].apply(count)
del keyword['Keyword']



# In[22]:


# Preparing training dataset...

# Merging training data with user, query, title, keyword & desc on appropriate keys to get data..

train = pd.merge(train, user,  on='UId')
train = pd.merge(train, query, on='QId')
train = pd.merge(train, title, on='TitleId')
train = pd.merge(train, desc,  on='DescId')
train = pd.merge(train, keyword, on='KeyId')

train.head()


# In[23]:


# Preparing testing dataset...

# Merging testing data with user, query, title, keyword & desc on appropriate keys to get data..

test = pd.merge(test, user,  on='UId')
test = pd.merge(test, query, on='QId')
test = pd.merge(test, title, on='TitleId')
test = pd.merge(test, desc,  on='DescId')
test = pd.merge(test, keyword, on='KeyId')

test.head()


# In[24]:


# Adding some useful features to train & test set which will be useful later.

# Since most of the features in the dataset are categorical features we will transform them into rate features...


# In[25]:


# A few helper methods...
train_avg_ctr = {}
train_avg_pctr = {}

def add(dataset, key, label, col, op):
    
    '''
        add a new feature 'label' to the 'dataset' using 'key' by applying operation 'op'.
    '''
    
    temp = dataset
    
    result = temp.groupby(key).agg([op])
    index  = result.index
    ctr    = result.get_values()
    
    temp = pd.DataFrame()
    temp[key] = index
    temp[label] = ctr
    if col=='CTR':
        train_avg_ctr[label]=np.mean(ctr)
    elif col == 'pCTR':
        train_avg_pctr[label]=np.mean(ctr)
        
    return temp


def addfeatures(dataset, train, keys, labels, col, op):
    
    '''
        addfeatures is used to add a set of features('labels') using 'keys' to the 'dataset'.
        The newly added features are constructed by applying 'op'(mean) on 'col' where 'col' is a 
        feature of the 'dataset'.
    '''

    for key in keys:
        temp = train[[key, col]]
        temp = add(temp, key, labels[key], col, op)
        dataset = pd.merge(dataset, temp, on=key, how='left')
    
    return dataset
    


# In[26]:


# For each categorical feature, we compute the average click-through rate as an additional one-dimensional feature. 

# Take AdID as an example. For each AdID, we com- pute the average click-through rate for all instances with the same AdID, 

# and use this value as a single feature. This feature represents the estimated click-through rate given its category. 

# We compute this kind of feature for AdURL, AdID, AdvertiserID, QueryID, KeywordID, TitleID, DescriptionID, 

#  UserID, DisplayURL, user’s age, user’s gender and (depth−position)/depth.  


# In[27]:


# Add Click Through Rate features to training set...

keys = ['AdURL', 'AdId', 'AdvId', 'Depth', 'Pos', 'QId', 'KeyId', 'TitleId', 'DescId', 'UId', 'Gender', 'Age','RPosition']
labels = {'AdURL':'mAdURL', 'AdId':'mAdCTR', 'AdvId':'mAdvCTR', 'Depth':'mDepthCTR', 'Pos':'mPosCTR', 'QId':'mQId', 
          'KeyId':'mKeyId', 'TitleId':'mTitleId', 'DescId':'mDescId', 'UId':'mUId', 'Gender': 'mGender', 'Age':'mAge','RPosition':'mRPosition'}

train_updated = addfeatures(train,train, keys, labels, 'CTR', 'mean')
train_updated.head()


# In[28]:


# Add Click Through Rate features to testing set...

test_updated = addfeatures(test,train, keys, labels, 'CTR', 'mean')

test_updated.head()


# In[29]:


# checking the null values in every column of test data
test_updated.isnull().sum()


# In[30]:


# we can observe
# 1. mAdURL
# 2. mAdCTR
# 3. mAdvCTR
# 4. mQId
# 5. mKeyId
# 6. mTitleId
# 7. mDescId
# 8. mUId
# these features are filled with NaN values while adding new features 


# In[31]:


# we will try to fill the NaN values with  the gloabal average of particular feature among whole train dataset
train_avg_ctr


# In[32]:


test_updated = test_updated.fillna(value=train_avg_ctr)


# In[33]:


# checking the null values in every column of test data after filling missing values
test_updated.isnull().sum()


# In[34]:


#  We generate pseudo-CTR features for AdID, AdvertiserID, QueryID, KeywordID, TitleID, DescriptionID, 

#  UserID, DisplayURL, user’s age, user’s gender and (depth−position)/depth. 


# In[35]:


# Add predicted Click Through Rate features to training set...

keys = ['AdURL', 'AdId', 'AdvId', 'Depth', 'Pos', 'QId', 'KeyId', 'TitleId', 'DescId', 'UId', 'Gender', 'Age','RPosition']
labels = {'AdURL':'pAdURL', 'AdId':'pAdCTR', 'AdvId':'pAdvCTR', 'Depth':'pDepthCTR', 'Pos':'pPosCTR', 'QId':'pQId', 
          'KeyId':'pKeyId', 'TitleId':'pTitleId', 'DescId':'pDescId', 'UId':'pUId', 'Gender': 'pGender', 'Age':'pAge','RPosition':'pRPosition'}

train_updated = addfeatures(train_updated,train, keys, labels, 'pCTR', 'mean')
train_updated.head()


# In[36]:


# checking the null values in every column of train data
train_updated.isnull().sum()


# In[37]:


# we will try to fill the NaN values with  the gloabal average of particular feature among whole train dataset
train_avg_pctr


# In[38]:


# Add predicted Click Through Rate features to testing set...

test_updated = addfeatures(test_updated,train, keys, labels, 'pCTR', 'mean')
test_updated.head()


# In[39]:


# checking the null values in every column of test data
train_updated.isnull().sum()


# In[40]:


test_updated = test_updated.fillna(value=train_avg_pctr)


# In[41]:


# Now that training & testing dataset are ready 


# In[42]:


# we will consider only pseduo features for both train and test

train_x = train_updated[['pAdURL','pAdCTR', 'pAdvCTR','pPosCTR','pQId', 
          'pKeyId', 'pTitleId', 'pDescId','pUId','pGender','pAge','pRPosition']].copy()

train_y = train_updated[['CTR']].copy()


# In[43]:


test_x = test_updated[['pAdURL','pAdCTR', 'pAdvCTR','pPosCTR','pQId', 
          'pKeyId', 'pTitleId', 'pDescId','pUId','pGender','pAge','pRPosition']].copy()

test_y = test_updated[['CTR']].copy()


# In[44]:


# Goal 2 ..performance metric


# 
# <font face = "Comic sans MS" size ="3" color = 'black'>
# <H3><u> Performance Metric:</u></H3><br>
# <ul>
# <li> AUC </li>
# <li> MAPE </li>
# </ul>
# <br>
# Question ?? How we calculate AUC here..!<br>
# 
# Lets try to understand- 
# <ul>
#  <li> The goal of the competition is to predict the click-through rate (#click / #impression) for each instance in the test set.</li>
#  <br>
#  <li> The goodness of the predictions is evaluated by the area under curve which is equivalent to the probability that a random pair of a positive sample (clicked ad) and a negative one (unclicked ad) is ranked correctly using the predicted click-through rate. </li>
#  <br>
#  <li> That is, an equivalent way of maximizing the AUC is to divide each instance into (#click) of positive samples and (#impression-#click) neg- ative samples, and then take #click on y-axis and #impressions on x-axis and calculate area under curve by applying trapezial rule for better approximation using the predicted click-through rate.</li></ul></font>
#                     
#                                                           /|
#                                                          / |
#                                                         /  |
#                                                         |  |
#                                                         |  |
#                                                         +--+
# <font face = "Comic sans MS" size ="3" color = 'black'>
# The area of this can be computed as average height * width = (height_left+height_right)/2 + width <br>
# =>  auc = (old_click_sum+click_sum) * no_click / 2.0  
# <br>
# The area is computed by cutting it into vertical slices at every change </font>
# 
# (Youtube link: https://www.youtube.com/watch?v=Yio4HbkQvkA)
# 
#   

# In[45]:


def scoreClickAUC(num_clicks, num_impressions, predicted_ctr):
    """
    Calculates the area under the ROC curve (AUC) for click rates

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    auc : the area under the ROC curve (AUC) for click rates
    """
     #sorting the values in descending order and store the index
    i_sorted = sorted(range(len(predicted_ctr)),key=lambda i: predicted_ctr[i],
                      reverse=True)
    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0

    # treat all instances with the same predicted_ctr as coming from the same bucket
    last_ctr = predicted_ctr[i_sorted[0]] + 1.0

    for i in range(len(predicted_ctr)):
        # when prev_ctr value at i-1 not match with current ctr value at i
        if last_ctr != predicted_ctr[i_sorted[i]]: 
            auc_temp += (click_sum+old_click_sum) * no_click / 2.0        
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]] #updating the last ctr value 
        # Calculating negative sample as #impressions - #clicks    
        no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        # Calculating postive samples as # of clicks 
        click_sum += num_clicks[i_sorted[i]]
    auc_temp += (click_sum+old_click_sum) * no_click / 2.0
    auc = auc_temp / (click_sum * no_click_sum) # That is the scaling to a total area of 1
    return auc


# In[46]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """
    return: MAPE 
    
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[49]:


import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
# depth of tree
for i in [2,4]:
    # number of estimators
    for j in [500,800,1000]:
    
        regr = xgb.XGBRegressor(max_depth= i, min_samples_split= 2,learning_rate=0.01,n_estimators = j, verbose=1)
        regr.fit(train_x, train_y)
        y_pred_test = regr.predict(test_x)
        y_pred_train = regr.predict(train_x)
        print('-'*50)
 
        roc_auc = scoreClickAUC(train['clicks'],train['impressions'],y_pred_train)
        err = mean_absolute_error(y_pred_train, train_y)
        print("the train auc with depth = ",i," and with esimators = ",j," is ", roc_auc)
        print("Train_MAPE", err * 100)

        roc_auc = scoreClickAUC(test['clicks'],test['impressions'],y_pred_test)
        err = mean_absolute_error(y_pred_test, test_y)
        print("the test auc with depth = ",i," and with esimators = ",j," is ", roc_auc)
        print("Test_MAPE",err * 100)


# In[63]:


import matplotlib.pyplot as plt
import numpy as np
regr = xgb.XGBRegressor(max_depth= 2, min_samples_split= 2,learning_rate=0.01,n_estimators = 800 , verbose=1)
regr.fit(train_x, train_y)
y_pred_test = regr.predict(test_x)
y_pred_train = regr.predict(train_x)
print('-'*50)


# In[62]:


roc_auc = scoreClickAUC(test['clicks'],test['impressions'],y_pred_test)
print(roc_auc)


# In[71]:


import matplotlib.pyplot as plt
#plt.title('Receiver Operating Characteristic')
tpr = test['clicks']
fpr = test['impressions'] - test['clicks']
# sorting data frame by name 
tpr = tpr.sort_values()
fpr = fpr.sort_values()

plt.plot(tpr, fpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.ylabel('# Positive samples')
plt.xlabel('# Negative samples')
plt.show() 


# In[74]:


from matplotlib import pyplot
# feature importance
print(regr.feature_importances_)


# In[73]:


from xgboost import plot_importance
plot_importance(regr)
pyplot.show()


# In[ ]:




