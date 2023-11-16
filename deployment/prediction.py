import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json

def run():
    # Load Model Classification
    with open('adaboost_logreg_best.pkl', 'rb') as file_1:
        classification_model = pickle.load(file_1)
    
    # Load Model Clustering
    with open('kp.pkl','rb') as file_2:
        clustering_model = pickle.load(file_2)

    # Load Clustering Scaler
    with open('scaler.pkl','rb') as file_3:
        scaler = pickle.load(file_3)
    
    # Load Clustering Numerical
    with open('num_col.txt','rb') as file_4:
        num_col = pickle.load(file_4)

    # Load Clustering Categorical
    with open('cat_col.txt','rb') as file_5:
        cat_col = pickle.load(file_5)

    # Choice of input: Upload or Manual Input
    inputType = st.selectbox("How would you like to input data ?", ["Upload Excel or CSV File", "Manual Input"])

    # Create Function for Prediction
    def predictData(df):
        # Classification prediction
        y_pred_uploaded = classification_model.predict(df)
        df['churn'] = y_pred_uploaded

        # Filter the DataFrame for Predicted Churn (1) 
        df_churn = df[df['churn'] == 1]
        
        churnCustomer = len(df_churn)

        if churnCustomer == 0:
            st.write('## There is no Customer predicted as Churn from this Data!')
        else:
            # Clustering prediction for Predicted Churn (1)
            ## Split Numerical and Categorical for K-Prototype
            data_cluster_num = df_churn[num_col]
            data_cluster_cat = df_churn[cat_col]

            ## Scale Numerical column
            num_scaled = scaler.transform(data_cluster_num)

            ## Merge Scaled Numerical + Categorical
            data_cluster_final = np.concatenate([num_scaled, data_cluster_cat], axis=1)
            data_cluster_final = pd.DataFrame(data_cluster_final, columns=['tenure', 'monthly_charges'] + cat_col)
            data_cluster_final = data_cluster_final.infer_objects()

            ## Mark Categorical Column
            index_cat_columns = [data_cluster_final.columns.get_loc(col) for col in cat_col] 

            ## Predict Cluster
            y_cluster = clustering_model.predict(data_cluster_final, categorical=index_cat_columns)
            # y_cluster = []
            #for rd in range(0, len(df_churn)): y_cluster.append(random.randint(0, 2)) # Random Generator for testing
            df_churn['cluster'] = y_cluster
            
            # Split Data into 3 Cluster DataFrames
            df_cluster_0 = df_churn[df_churn['cluster'] == 0]
            df_cluster_1 = df_churn[df_churn['cluster'] == 1]
            df_cluster_2 = df_churn[df_churn['cluster'] == 2]

            st.write('## Result')
            st.write('##### Here are some suggestion to minimalize churn potential for each customer')
            c0, c1, c2 = '', '', ''
            for x in df_cluster_0['name']: c0 += str(x) + ', '
            for y in df_cluster_1['name']: c1 += str(y) + ', '
            for z in df_cluster_2['name']: c2 += str(z) + ', '
            
            suggestion_0 = '''
                - Menawarkan paket dengan tambahan kecepatan selama 3 bulan bagi yang telah berlangganan di atas 3 tahun
                - Membuka seluruh channel TV saat event hari besar seperti lebaran, natal dan lain lain
                - Memberikan penawaran khusus untuk meningkatkan kecepatan internet kepada mereka
            '''

            suggestion_1 = '''
                - Memberikan penawaran dengan banyak keuntungan jika berlangganan untuk jangka panjang 
                - Menawarkan paket internet DSL tahunan dengan harga yang terjangkau
            '''

            suggestion_2 = '''
                Memberikan paket khusus dengan kriteria sebagai berikut :
                - Kecepatan tinggi tetapi banwidth lebih rendah dengan harga yang lebih murah dari paket normal
                - Kecepatan rendah tetapi banwidth besar sehingga koneksi jauh lebih stabil dengan harga yang lebih murah dari paket normal
            '''

            if c0 != '':
                st.write('Suggestion for `', c0[0:-2], '` is')
                st.write(suggestion_0)
                st.markdown('---')
            
            if c1 != '':
                st.write('Suggestion for `', c1[0:-2], '` is')
                st.write(suggestion_1)
                st.markdown('---')
            
            if c2 != '':
                st.write('Suggestion for `', c2[0:-2], '` is')
                st.write(suggestion_2)
                st.markdown('---')
    
    def tenureMonthToYear():
        year = st.session_state.tenurem % 12
        if year == 0:
            st.session_state.tenurey = int((st.session_state.tenurem / 12))
        else:
            st.session_state.tenurey = int((st.session_state.tenurem / 12) + 1)
    
    def calculateChargesAndCategory():
        st.session_state.tcharges = int((st.session_state.mcharges * st.session_state.tenurem))
        if st.session_state.mcharges <= 30:
            st.session_state.catcharges = 'Low Expense'
        elif st.session_state.mcharges <= 60:
            st.session_state.catcharges = 'Medium Expense'
        elif st.session_state.mcharges <= 90:
            st.session_state.catcharges = 'Medium High Expense'
        else:
            st.session_state.catcharges = 'High Expense'

    # A. For CSV
    if inputType == "Upload Excel or CSV File":
        uploaded_file = st.file_uploader("Choose a Excel or CSV file", type=["csv", "xlsx"], accept_multiple_files=False)
        if uploaded_file is not None:
            split_file_name = os.path.splitext(uploaded_file.name)
            # file_name = split_file_name[0]
            file_extension = split_file_name[1]

            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            else:    
                df = pd.read_excel(uploaded_file)
            # st.dataframe(df.head())
            predictData(df)
    # B. For Manual        
    else:
    # Create Form
        # with st.form(key='Form Parameters'):
        name = st.text_input('Name', value='', help='Customer Name')

        col_left, col_mid, col_right = st.columns([3, 2, 2])
        gender =  col_left.selectbox('Gender', ('Male', 'Female'), index=0)
        with col_mid:
            tenure =  st.number_input('Tenure (Month)', min_value=1, max_value=999, step=1, help='Month', key='tenurem', on_change=tenureMonthToYear)
        with col_right:
            tenure_year = st.number_input('Tenure (Year)', min_value=1, max_value=999, step=1, disabled=True, key='tenurey')
        
        col1, col2, col3 = st.columns([1, 1, 1])
        senior_citizen = col1.radio(label='Senior Citizen?', options=['Yes', 'No'], help='Choose \'Yes\' for 61 years old above')
        partner = col2.radio(label='Having a partner?', options=['Yes', 'No'])
        dependents = col3.radio(label='Having a dependents?', options=['Yes', 'No'], help='For example : children')
        
        # col4, col5 = st.columns([1, 1])
        # internet_service =  col4.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'), index=0)

        col4, col5, col6 = st.columns([1, 1, 1])
        internet_service =  col4.radio(label='Subs for Phone service?', options=['DSL', 'Fiber optic', 'No'])
        phone_service = col5.radio(label='Subs for Phone service?', options=['Yes', 'No'])
        multiple_lines = col6.radio(label='Subs for Multiple Lines?', options=['Yes', 'No', 'No Phone Services'])

        col7, col8, col9 = st.columns([1, 1, 1])
        online_security = col7.radio(label='Subs for Online Security?', options=['Yes', 'No', 'No Internet Services'])
        online_backup = col8.radio(label='Subs for Online Backup?', options=['Yes', 'No', 'No Internet Services'])
        device_protection = col9.radio(label='Having Device Protections?', options=['Yes', 'No', 'No Internet Services'])
        tech_support = col7.radio(label='Having Tech Support service?', options=['Yes', 'No', 'No Internet Services'])
        streaming_tv = col8.radio(label='Subs for TV Streaming?', options=['Yes', 'No', 'No Internet Services'])
        streaming_movies = col9.radio(label='Subs for Movie Streaming?', options=['Yes', 'No', 'No Internet Services'])

        col_pm1, col_pm2, col_pm3 = st.columns([3, 3, 2])
        contract = col_pm1.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'), index=0)
        payment_method = col_pm2.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), index=0)
        paperless_billing = col_pm3.selectbox('Paperless billing?', ('Yes', 'No'), index=0)

        col_charges1, col_charges2, col_charges3 = st.columns([1, 1, 2])
        monthly_charges =  col_charges1.number_input('Monthly Charges', min_value=1, max_value=999, step=1, help='Amount to paid per month', key='mcharges', on_change=calculateChargesAndCategory)
        total_charges = col_charges2.number_input('Total Charges', min_value=1, max_value=999999, step=1, disabled=True, key='tcharges')
        charges_cat = col_charges3.text_input('Chargest Category', disabled=True, key='catcharges')

        # st.button('Predict', on_click=predict)
        data_inf = {
            'name': name, 
            'gender': gender, 
            'senior_citizen': senior_citizen, 
            'partner': partner, 
            'dependents': dependents, 
            'tenure': int(tenure), 
            'phone_service': phone_service, 
            'multiple_lines': multiple_lines, 
            'internet_service': internet_service, 
            'online_security': online_security, 
            'online_backup': online_backup, 
            'device_protection': device_protection, 
            'tech_support': tech_support, 
            'streaming_tv': streaming_tv, 
            'streaming_movies': streaming_movies, 
            'contract': contract, 
            'paperless_billing': paperless_billing, 
            'payment_method': payment_method, 
            'monthly_charges': monthly_charges, 
            'total_charges': int(total_charges), 
            'monthly_charges_cat': charges_cat, 
            'tenure_year': tenure_year
            }

        if st.button('Predict'):
            data_inf = pd.DataFrame([data_inf])
            # st.dataframe(data_inf.head())
            predictData(data_inf)

if __name__ == '__main__':
    run()