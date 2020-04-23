# Plotly-project

Data visualization with Plotly


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn import feature_selection

import statsmodels.api as sm

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

import chart_studio.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

Data1 = pd.read_csv("timesData.csv")

Data1.head() 

Data=pd.DataFrame(Data1)

#Since its a raw dataset, we need to clean the data and remove or fill the null values 

Data.isnull().sum()

#Forward fill the missing values

Data.ffill(axis = 0) 

Data=Data.dropna()

# CHOROMAP


data = dict(type='choropleth',
           colorscale = 'Portland',
            locations = Data['country'],
            z = Data['world_rank'],
            locationmode = 'country names',
            text = Data['country'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"College rankings"}
            )
layout = dict(title = 'Countries with the best colleges',
              geo = dict(scope='world'))
              

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# LINE CHART

df = Data.iloc[:100,:]

import plotly.graph_objs as go

trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.university_name)

trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
                    
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
             
fig = dict(data = data, layout = layout)
iplot(fig)

# SCATTER PLOT

df2014 = Data[Data.year == 2014].iloc[:100,:]

df2015 = Data[Data.year == 2015].iloc[:100,:]

df2016 = Data[Data.year == 2016].iloc[:100,:]


import plotly.graph_objs as go

trace1 =go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2014.university_name)

trace2 =go.Scatter(
                    x = df2015.world_rank,
                    y = df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2015.university_name)

trace3 =go.Scatter(
                    x = df2016.world_rank,
                    y = df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2016.university_name)
                    
data = [trace1, trace2, trace3]

layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)
             )
             
fig = dict(data = data, layout = layout)

iplot(fig)

# PI CHART


df2016 = Data[Data.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  

labels = df2016.university_name

fig = {"data": [{"values": pie1_list,"labels": labels,"domain": {"x": [0, .5]},"name": "Number Of Students Rates","hoverinfo":"label+percent+name",
      "hole": .3,"type": "pie"},],
  "layout": {"title":"Universities Number of Students rates","annotations": [{ "font": { "size": 20},"showarrow": False,"text": "Number of Students",
  "x": 0.20,"y": 1},]}}
  
iplot(fig)

# SCATTER MATRIX

import plotly.figure_factory as ff

dataframe = Data[Data.year == 2015]

data2015 = dataframe.loc[:,["research","international_students", "total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)

# MULTIPLE SUBPLOTS

trace1 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.research,
    name = "research"
)

trace2 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.citations,
    xaxis='x2',
    yaxis='y2',
    name = "citations"
)

trace3 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.income,
    xaxis='x3',
    yaxis='y3',
    name = "income"
)

trace4 = go.Scatter(
    x=dataframe.world_rank,
    y=dataframe.total_score,
    xaxis='x4',
    yaxis='y4',
    name = "total_score"
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Research, citation, income and total score VS World Rank of Universities'
)


fig = go.Figure(data=data, layout=layout)

iplot(fig)


