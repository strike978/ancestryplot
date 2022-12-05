import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

# Reading the data from the csv file and storing it in a dataframe.
st.set_page_config(
    page_title="Data - ancestryplot")


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('data.csv')
    return (df)


df = load_data()

df['Population'] = df['Population'].str.split(':').str[-1]
selected_group = st.selectbox('Select a group', df.columns[4:])
df = df[df[selected_group] != 0]

df = df.sort_values(selected_group, ascending=False)

gb = GridOptionsBuilder.from_dataframe(
    df[['Population', 'Region', selected_group]])
gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
gridOptions = gb.build()
AgGrid(df[['Population', 'Region', selected_group]],
       fit_columns_on_grid_load=True, gridOptions=gridOptions)
