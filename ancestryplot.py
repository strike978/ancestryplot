import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import plotly.express as px
# import xlsxwriter
# import io
import kaleido
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Setting the title of the page and the icon of the page.
st.set_page_config(
    page_title="ancestryplot",
    page_icon="📈"
)

# A boolean variable that is used to show the data and the map when the user clicks on the button.
show_data = False
show_map = False


@st.cache(allow_output_mutation=True)
def load_data():
    # Loading the data from the csv file.
    df = pd.read_csv('data.csv')
    return (df)


# Splitting the string in the column `Population` by the character `:` and taking the last element of
# the resulting list.
@st.cache()
def convert_Population_column(df):
    df['Population'] = df['Population'].str.split(':').str[-1]
    return (df)


# Loading the data from the csv file and converting the column `Population` to a string.
df = load_data()
convert_Population_column(df)

# Creating a title and a description for the page.
st.write("# 📈ancestryplot")

st.markdown(
    """
Create models based on genetic data from mesolithic and neolithic populations. With our easy-to-use interface, you can quickly and easily plot maps and data tables to visualize ancestry information for modern
populations. 
Start creating your own ancestry models today!
"""
)

modification_container = st.container()

# Based on https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
# Creating a container with the title "Filter by" and the columns `df.columns[1]` and `df.columns[4]` selected by default.

with modification_container:
    to_filter_columns = st.multiselect("Filter by", (df.loc[:, ~df.columns.isin([df.columns[0], df.columns[2], df.columns[3]])]).columns,
                                       default=[df.columns[1], df.columns[4]])
    l = []
    for i, column in enumerate(to_filter_columns):
        df = df[df[column] != 0]
        left, right = st.columns((1, 20))
        left.write("↳")
        # Treat columns with < 10 unique values as categorical
        if is_numeric_dtype(df[column]):
            _min = float(df[column].min())
            _max = float(df[column].max())
            step = (_max - _min) / 100
            user_num_input = right.slider(
                f"{column}",
                _min,
                _max,
                (_min, _max),
                step=step,
            )
            df = df[df[column].between(*user_num_input)]
        else:
            user_cat_input = right.multiselect(
                f"{column}",
                df[column].unique(),
                default=list(df[column].unique()),
            )
            df = df[df[column].isin(user_cat_input)]
        l.append(column)

# Creating a text input with the label "Model" and the placeholder "Enter a model name".
model_name = st.text_input("Model", placeholder="Enter a model name", value='')


# Checking if the user entered a model name.
if model_name != '':
    # Creating a new column in the dataframe `df` with the name `model_name` and the values of the column
    # are the sum of the values of the columns in the list `l`.
    df[model_name] = df[l].sum(axis=1, numeric_only=True)
    # Sorting the dataframe by the column `model_name` in descending order.
    df = df.sort_values(model_name, ascending=False)

    with st.expander("🗃 Data", expanded=True):
        # Creating a download button for the dataframe.
        col1, _, col3 = st.columns([3, 6, 3])
        with col1:
            if st.button("🗃 Show Data"):
                with col3:
                    st.download_button(label="📥 Export to CSV", data=df.loc[:, ['Population', 'Region', model_name]].to_csv(
                        index=False), file_name=f'{model_name}.csv', mime='text/csv')
                show_data = True

            # with col3:
            #     pass
                # buffer = io.BytesIO()
                # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                #     df.loc[:, ['Population', 'Region', model_name]
                #            ].to_excel(writer, index=False)
                #     writer.save()
                # st.download_button(
                #     label="📥 Export to Excel",
                #     data=buffer,
                #     file_name=f"{model_name}.xlsx",
                #     mime="application/vnd.ms-excel"
                # )

       # Creating a table with the data.
        if show_data:
            gb = GridOptionsBuilder.from_dataframe(
                df.loc[:, ['Population', 'Region', model_name]])
            gb.configure_pagination(
                paginationAutoPageSize=False, paginationPageSize=15)
            gridOptions = gb.build()
            AgGrid(df.loc[:, ['Population', 'Region', model_name]],
                   fit_columns_on_grid_load=True, gridOptions=gridOptions)

    # Creating a map with the data.
    with st.expander("🗺 Map", expanded=True):
     # Checking if the user selected more than one column.
        if len(l) < 2:
            st.write("Please select more than one column")
        else:
            col1, _, col3 = st.columns([3, 6, 3])
            with col1:
              # Creating a map with the data.
                if st.button("🗺 Show Map"):
                    with st.spinner('Loading...'):
                        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
                                                zoom=1, hover_name='Population', color=model_name, size=model_name,
                                                mapbox_style='open-street-map', color_continuous_scale="bluered", opacity=1, size_max=15)
                        fig.update_layout(mapbox_style="open-street-map")
                        with col3:
                            # Creating a download button for the image.
                            btn = st.download_button(
                                label="📥 Download Map",
                                data=fig.to_image(
                                    format="png", engine="kaleido"),
                                file_name=f"{model_name}.png",
                                mime="image/png"
                            )
                        show_map = True
            if show_map:
                st.plotly_chart(fig)
