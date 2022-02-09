#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install --upgrade dash
#!pip install --upgrade jupyter_dash


# In[2]:


import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as gp

from jupyter_dash import JupyterDash
import dash
from dash import Dash, dcc, html, Input, Output
from dash.exceptions import PreventUpdate


# In[3]:


def extractdata():
    df = pd.read_csv('01_Tables/00_Clean_Offence.csv', dtype='unicode')
    # Extract repeated offenders
    df = df[['Driving_License_No','Gender', 'Age', 'Name']].copy() 
    df.dropna(subset=['Driving_License_No'], inplace = True)
    df['Driving_License_No'] = df['Driving_License_No'].str.upper()
    # to check number of offences. 
    lldict =df.groupby('Driving_License_No')[['Driving_License_No']].count()
    lldict.rename(columns = {'Driving_License_No':'Num_Offences'},inplace = True)
    lldict.reset_index(inplace = True)
    dict1 = dict(zip(lldict.Driving_License_No, lldict.Num_Offences))
    dln = df['Driving_License_No'].copy()
    df['Num_Offences'] = dln.map(dict1)
    df = df[df['Num_Offences']>1] # select only repeated offenders
    df = df.drop_duplicates(subset=['Driving_License_No'])
    #df1['Gender']= df1['Gender'].fillna('Unknown')
    df['Age'] = df['Age'].astype(float)
    df['age_groups'] = pd.cut(df['Age'], bins=[0,14,19, 24, 29, 34, 39,44, 49, 54, 59, 64,69,74, np.inf])
    
    
    df1_2 = df.groupby(['Num_Offences','Gender'])[['Driving_License_No']].count()
    df1_2.reset_index(inplace = True)
    df1_2.rename(columns = {'Driving_License_No': 'Num_Offenders'},inplace = True) 
    ggroup=df1_2.groupby(['Num_Offences'])
    df1_2['Percentage'] = ggroup['Num_Offenders'].apply(lambda x: 100 * x/float(x.sum())).values
    df1_2 = ggroup.apply(lambda x: x.sort_values(by = ['Gender'], ascending = False))
    df1_2.reset_index(drop = True, inplace = True)   
    
    dft = df.groupby(['Num_Offences']).count()[['Driving_License_No']].nlargest(4,'Driving_License_No')
    dft.reset_index(inplace = True)
    dft.rename(columns = {'Driving_License_No': 'Num_Offenders'},inplace = True) 
    df3 = df.groupby(['Num_Offences', 'Name']).count()[['Driving_License_No']]
    df3.reset_index(inplace = True)
    df3.rename(columns = {'Driving_License_No': 'Num_Offenders'},inplace = True) 
    df3 = df3.groupby('Num_Offences', sort = False)
    df3 = df3.apply(lambda x: x.sort_values(by = ['Num_Offenders'], ascending = False))
    df3 = df3.reset_index(drop=True)
    df3 = df3.groupby('Num_Offences').head(5)
    df3 = df3[df3['Num_Offences'].isin(dft['Num_Offences'])]
    
    df4 = df.groupby(['age_groups'])[['Driving_License_No']].count()
    df4.rename(columns = {'Driving_License_No':'Num_Offenders'},inplace = True)
    df4.reset_index(inplace = True)
    df4['age_groups'] = df4['age_groups'].astype(str)
    
    df5 = df.groupby(['age_groups', 'Name']).count()[['Driving_License_No']]
    df5.reset_index(inplace = True)
    df5.rename(columns = {'Driving_License_No': 'Num_Offenders'},inplace = True) 

    df5 = df5.groupby('age_groups', sort = False)
    df5 = df5.apply(lambda x: x.sort_values(by = ['Num_Offenders'], ascending = False))
    df5 = df5.reset_index(drop=True)

    ag = list(df5['age_groups'].unique())[2:10]
    df5 = df5[df5['age_groups'].isin(ag)]
    df5 = df5.groupby('age_groups').head(1)

    return [df, df1_2, df3, df4, df5]


# In[4]:


def drawfig():
    [df, df1_2, df3, df4, df5]=extractdata()

    fig1 = px.bar(df1_2, x='Num_Offences', y='Num_Offenders', color ='Gender', 
                  color_discrete_sequence=["blue", "magenta"], hover_data = ['Num_Offenders'],
                  barmode = 'stack', text = df1_2['Num_Offenders'])
    fig1.update_layout(xaxis = dict(tickvals = df1_2['Num_Offences'],ticktext = df1_2['Num_Offences'],                                      
                        title = 'Number of offences',title_font_size = 14), 
                       yaxis_title = 'Number of Repeated Offenders')
    fig1.update_layout(title={'text' : 'Number of Repeated Offenders vs. Number of offences',
                            'x':0.45,'xanchor': 'center'})

    fig2 = px.bar(df1_2, x='Num_Offences', y='Percentage', color ='Gender', 
                  color_discrete_sequence=["blue", "magenta"], hover_data = ['Num_Offenders'],
                  barmode = 'stack', text = df1_2['Percentage'].apply(lambda s:'{:,.0f}%'.format(s)))
    fig2.update_layout(xaxis = dict(tickvals = df1_2['Num_Offences'],ticktext = df1_2['Num_Offences'],                                      
                        title = 'Number of offences',title_font_size = 14), 
                       yaxis_title = 'Percentage of offenders')
    fig2.update_layout(title={'text' : 'Gender Percentage Distribution of Repeated Offenders',
                            'x':0.45,'xanchor': 'center'})

    fig3 = px.bar(df3, y="Name", x = 'Num_Offenders', color='Num_Offences', orientation = 'h',
                  color_discrete_sequence=["blue", "magenta", "yellow", "red"],
            labels = {'Name': 'Type of Offences', 'Num_Offenders':'Number of Repeated Offenders'})
    fig3.update_layout(title={'text' : 'Major offences by Repeated Offenders','x':0.45,'xanchor': 'center'})
    fig3.update_layout(showlegend=False)
    fig3.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))


    fig4 = px.bar(df4, x='age_groups', y='Num_Offenders', color ='age_groups', 
                  hover_data = ['Num_Offenders'],barmode = 'stack', text = df4['Num_Offenders'])
    fig4.update_layout(xaxis = dict(tickvals = df4['age_groups'],ticktext = df4['age_groups'],                                      
                        title = 'Age Group',title_font_size = 14), 
                       yaxis_title = 'Number of offenders')
    fig4.update_layout(title={'text' : 'Number of Repeated Offenders vs. Age Group',
                            'x':0.45,'xanchor': 'center'})

    fig5 = px.sunburst(df5, path=['age_groups','Name'], values='Num_Offenders',color = 'Name')
    fig5.update_layout(title={'text' : 'Prominent Offence of Repeated Offenders by Age Group',
                              'x':0.45,'xanchor': 'center'})
    
    return [fig1, fig2, fig3, fig4, fig5]


# In[5]:


app = JupyterDash(__name__)
server = app.server
app.title="Druk-BSB"


# In[6]:


tab_style = {
    "background": "#4747c4",
    'text-transform': 'uppercase',
    'color': 'white',
    'border': 'grey',
    'font-size': '20px',
    'font-weight': 400,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '10px',
    'padding':'24px'    
}

tab_selected_style = {
    "background": "#fffff7",
    'text-transform': 'uppercase',
    'color': 'black',
    'font-size': '20px',
    'font-weight': 1000,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '10px',
    'padding':'24px',
    
}


# In[7]:


app.layout = html.Div([
    html.Div([
        html.H1('Interactive Dashboard for Repeated Traffic Offences',style={'textAlign':'center','color':'#FC0A02','fontsize':50}),
       
    ], id='header_div'),
        
     html.Div([
        dcc.Tabs(id="tabs", value='active', children=[
            dcc.Tab(label='Home', value='home', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Age-based Analysis', value='age',style=tab_style, selected_style=tab_selected_style)
        ]),
    ], id='tabs_div'),
   
        
    #Image displace
    html.Div([
        # output graphic (plot1)
        dcc.Graph(id='plot1'),
        # output graphic (plot2)
        dcc.Graph(id='plot2')  
    ], style = {'display': 'flex'}),        
    
], id='main-div')


# In[8]:


@app.callback([    
    Output('plot1', 'figure'),
    Output('plot2', 'figure')
],
    Input('tabs','value')
)

def update_output(tab): 
    
    [fig1, fig2, fig3, fig4, fig5] = drawfig()      
    if(tab =='home'):
        return [fig1, fig3]
    elif(tab == 'age'):
        return [fig4,fig5]
    else:
        return [fig1, fig3]


# In[9]:


app.run_server()


# In[ ]:




