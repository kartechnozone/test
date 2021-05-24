#Basic libraries
import pandas as pd 
import numpy as np 
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
#%matplotlib inline

highlights = pd.read_csv('highlights.csv',parse_dates=['time'])
highlights.head()
def f(row):
    
    '''This function returns sentiment value based on the overall ratings from the user'''
    
    if row['highlights'] == 0:
        val = 'dot'
    elif row['highlights'] == 1:
        val = 'one'
    elif row['highlights'] == 2:
        val = 'two'
    elif row['highlights'] == 4:
        val = 'four'
    elif row['highlights'] == 6:
        val = 'six'
    elif row['highlights'] == -1:
        val = 'out'
    elif row['highlights'] == -2:
        val = 'extras'
    elif row['highlights'] == -3:
        val = 'break'
    else:
        val = -3
    return val

#Applying the function in our new column
highlights['activity'] = highlights.apply(f, axis=1)
highlights.head()


plt.figure(1)
plt.subplot(1, 2, 1)
sns.lineplot(data=highlights, hue='activity', x='time', y='FanEngageScore')
plt.title('time vs fanenagement score[highlights')

plt.subplot(1, 2, 2)
sns.lineplot(data=highlights, hue='activity', x='time', y='HRCount')
plt.title('time vs  heart rate count[highlights')

plt.figure(2)
plt.subplot(1, 2, 1)
sns.lineplot(data=highlights, hue='activity', x='time', y='FanEngageScore')
plt.title('time vs fanenagement score[highlights')

plt.subplot(1, 2, 2)
sns.lineplot(data=highlights, hue='activity', x='time', y='sentiment')
plt.title('time vs  sentiment[highlights')
plt.show()