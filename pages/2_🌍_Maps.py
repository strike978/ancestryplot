import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(
    page_title="Maps - ancestryplot",
    layout='wide')


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('data.csv')
    return (df)


df = load_data()

df['Population'] = df['Population'].str.split(':').str[-1]
selected_group = st.selectbox('Select a group', df.columns[4:])
df = df[df[selected_group] != 0]

fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
                        zoom=1, hover_name='Population', color=selected_group, size=selected_group,
                        mapbox_style='open-street-map', color_continuous_scale="bluered", opacity=1, size_max=15)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=True)
