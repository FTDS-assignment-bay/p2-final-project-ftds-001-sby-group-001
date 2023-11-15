import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def run():
    #Show dataframe
    # st.title('Data Overview')
    df = pd.read_csv('telco_data_clean.csv')
    # st.dataframe(df.head())

    st.title('Exploratory Data Analysis')
    plot_selection = st.selectbox(label='Choose', 
                                  options=['Customer Distribution', 
                                           'Top Total Churn City',
                                           'Customers reasons for churning', 
                                           'Churn Reason', 
                                           'Age Distribution Churn vs Stayed',  
                                           'Gender Distribution Churn vs Stayed'])
    
    # Plot 1
    def plot_1():
        st.write('#### Pie Chart for Customer Status Distribution')
        # fig_1 = plt.figure()
        # customer_status_count = df['Customer Status'].value_counts()
        # fig_1, ax = plt.subplots()
        # ax.pie(customer_status_count, labels=customer_status_count.index, autopct='%1.1f%%')
        # ax.set_title('Customer Status Distribution')
        # st.pyplot(fig_1)
        # with st.expander('Explanation'):
        #     st.text('''
        #         The data frame indicates that 26.5% of customers have churned. 
        #         The "Joined" Category row will be removed, as it doesn't offer 
        #         any useful insights into the churn rate.
        #     ''')
        st.text('''
                The data frame indicates that 26.5% of customers have churned. 
                The "Joined" Category row will be removed, as it doesn't offer 
                any useful insights into the churn rate.

                Sisa ne nyusul
            ''')

    # st.write('## Histogram Limit Balance')
    # fig = plt.figure(figsize=(15,5))
    # sns.histplot(df['limit_balance'], bins=20, kde=True).set(title='limit_balance')
    # st.pyplot(fig)
    # st.write('Based on histogram, column _limit\_balance_ skewness is positive, meaning the data distribution is not normal.')
    # st.markdown('---')

    # st.write('## Average Amount of Bill Statement')
    # df_amt = pd.DataFrame()
    # bill_amt = []
    # pay_amt = []
    # for i in range(1, 7):
    #     bill_amt.append(df['bill_amt_' + str(i)].mean())
    #     pay_amt.append(df['pay_amt_' + str(i)].mean())
    # df_amt['bill_amt'] = bill_amt
    # df_amt['pay_amt'] = pay_amt
    # fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # axis_label = sns.barplot(ax=ax[0], data=df_amt, x=df_amt['bill_amt'].index, y=bill_amt, orient='v')
    # ax[0].set_title('Bill Statement')
    # axis_label = sns.barplot(ax=ax[1], data=df_amt, x=df_amt['pay_amt'].index, y=pay_amt, orient='v')
    # ax[1].set_title('Payment')
    # st.pyplot(fig)
    # st.write('Average amount of bill statement is decrease every month, showing people using their credit card less during this period.')
    # st.markdown('---')

    # st.write('## Barplot Sex')
    # fig = plt.figure(figsize=(10,5))
    # sns.countplot(x='sex', data=df)
    # st.pyplot(fig)
    # st.write('Most of the bank customer is female')
    # st.markdown('---')

    # st.write('## Barplot Marital Status')
    # fig = plt.figure(figsize=(15,5))
    # sns.countplot(x='marital_status', data=df)
    # st.pyplot(fig)
    # st.write('Most of the bank customer is married')
    # st.markdown('---')

    if plot_selection == "Customer Distribution":
        plot_1()

if __name__ == '__main__':
    run()