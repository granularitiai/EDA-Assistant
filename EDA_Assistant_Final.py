#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install langchain-openai


# In[1]:


import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import tabulate
import pyarrow
import openai


# In[2]:


openai_api_key = 'Your OpenAI API key'


# In[19]:


st.set_page_config(page_title='📊 EDA Assistant')
st.title('📊 EDA Assistant')
st.header('To use this app you will need to have an OpenAI key, please use this site  to sign up to recieve a key')


# In[20]:


def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df


# In[21]:


def generate_response(csv_file, input_query):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    df = load_csv(csv_file)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    response = agent.run(input_query)
    return st.success(response)


# In[22]:


uploaded_file = st.file_uploader('Upload a CSV file', type = ['csv'])
question_list = [
    'What are the basic characteristics of my dataset?',
    'What is the overall structure of my dataset?',
    'What patterns exist in the data?',
    'Are there any outliers present?',
    'What are the missing values in the data set?',
    'Is there any correlation between variables?',
    'Are there any discrepancies between observed values and expected values?',
    'Do I need to transform any variables before analysis?',
    'Other:']
query_text = st.selectbox('Select an example query:', question_list, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not(uploaded_file and query_text))


# In[23]:


if query_text == 'Other:':
    query_text = st.text_input('Enter your query:', placeholder = 'Enter query here ...', disabled = not uploaded_file)
if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API Key!')
if openai_api_key.startswith('sk-') and (uploaded_file is not None):
    st.header('Output')
    generate_response(uploaded_file, query_text)


# In[ ]:





# In[ ]:




