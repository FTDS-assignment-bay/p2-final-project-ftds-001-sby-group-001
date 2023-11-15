import streamlit as st
import eda
import prediction

# navigation = st.sidebar.selectbox('Select Page :', ('EDA', 'Predict Credit Card Default'))

# if navigation == 'EDA':
#     eda.runEDA()
# else:
#     prediction.runPredictor()

# Set page title and icon
# st.set_page_config(page_title='Final Project', page_icon='⭐')

# Create sidebar navigation
st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] {{
            background-image: url(https://raw.githubusercontent.com/FTDS-assignment-bay/FTDS-007-HCK-group-002/main/assets/ChurnGuardian-Logo-Transparants.png);
            background-repeat: no-repeat;
            padding-top: 20px;
            background-position: 10px 50px;
            background-size: 310px 85px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

selected_page = st.sidebar.radio('Select Page', ('📋 Home Page', '📊 Exploratory Data Analysis', '💻 Model'))
#streamlit run app.py