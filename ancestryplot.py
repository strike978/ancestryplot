import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode


# Setting the page title, icon, layout and initial sidebar state.
st.set_page_config(
    page_title="ancestryplot",
    page_icon="🧬",
    layout='wide',
    initial_sidebar_state="expanded"
)


st.write("# Welcome to Ancestryplot! 🧬")


st.markdown(
    """
    View maps and data for world populations! Genetic admixtures of the current human populations are featured on every map.
    **👈 Select a page from the sidebar** to see some examples
    of what Ancestryplot can do!
    ### Explore and discover genetic ancestry data. Search for populations that were most likely to have a common ancestor, or for populations that were both common (e.g., the population of the human family) and that were in close contact.
"""
)

with st.sidebar:
    pass
