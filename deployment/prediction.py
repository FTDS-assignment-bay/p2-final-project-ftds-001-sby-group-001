import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import json

def run():
    # Load Model
    with open('adaboost_logreg_best.pkl', 'rb') as file_1:
        classification_model = pickle.load(file_1)
    
    # with open('kmeans_best.pkl', 'rb') as file_2:
    #     clustering_model = pickle.load(file_2)

    # Choice of input: Upload or Manual Input
    inputType = st.selectbox("How would you like to input data ?", ["Upload CSV File", "Manual Input"])
    
    # A. For CSV
    if inputType == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # st.dataframe(df.head(15))

            # Classification prediction
            y_pred_uploaded = classification_model.predict(df)
            df['churn_prediction'] = y_pred_uploaded

            # Filter the DataFrame for Predicted Churn (1) 
            df_churn = df[df['churn_prediction'] == 1]

            # Clustering prediction for Predicted Churn (1)
            # y_cluster = clustering_model.predict(df_churn)
            y_cluster = []
            for rd in range(0, len(df_churn)): y_cluster.append(random.randint(0, 2))
            df_churn['cluster'] = y_cluster
            
            # Split Data into 3 Cluster DataFrames
            df_cluster_0 = df_churn[df_churn['cluster'] == 0]
            df_cluster_1 = df_churn[df_churn['cluster'] == 1]
            df_cluster_2 = df_churn[df_churn['cluster'] == 2]

            st.write('## Result')
            st.write('##### Here are some suggestion to minimalize churn potential for each customer')
            c0, c1, c2 = '', '', ''
            for x in df_cluster_0['name']: c0 += x + ', '
            for y in df_cluster_1['name']: c1 += y + ', '
            for z in df_cluster_2['name']: c2 += z + ', '
            
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
            
    else:
        st.write('## sek durung cok, ngantuk aku')

    # # Create Form
    # with st.form(key='Form Parameters'):
    #     limit_balance = st.number_input('Limit Balance', min_value=0, max_value=1000000, step=5000)
    #     # age = st.number_input('Age', min_value=0, max_value=99, step=1, help='Usia')
    #     age_cat =  st.selectbox('Age Category', ('Children', 'Young Adult', 'Adult', 'Middle Age', 'Old Age'), index=0, help='1-16 : Children, 17-30 : Young Adult, 31-40 : Adult, 41-50 : Middle Age, 50+ Old Age')
    #     sex =  st.selectbox('Sex', ('Male', 'Female'), index=0)
    #     education_level = st.selectbox('Education Level', ('Graduate School', 'University', 'High School', 'Others', 'Unknown'), index=0)
    #     marital_status =  st.selectbox('Marital Status', ('Married', 'Single', 'Others'), index=0)
    #     st.markdown('---')
    #     pay_1 = st.selectbox('Repayment in September', ('-2', '-1', '0', '1', '2', '3', '4', '5', '6'), index=2)
    #     pay_2 = st.selectbox('Repayment in August', ('-2', '-1', '0', '1', '2', '3', '4', '5'), index=2)
    #     pay_3 = st.selectbox('Repayment in July', ('-2', '-1', '0', '1', '2', '3', '4'), index=2)
    #     pay_4 = st.selectbox('Repayment in June', ('-2', '-1', '0', '1', '2', '3'), index=2)
    #     pay_5 = st.selectbox('Repayment in May', ('-2', '0', '2', '3'), index=1)
    #     pay_6 = st.selectbox('Repayment in April', ('-2', '0','1', '2', '3', '4'), index=1)
    #     st.markdown('---')
    #     bill_amt_1 = st.number_input('Bill Amount1', min_value=-1000000, max_value=1000000, step=50000)
    #     bill_amt_2 = st.number_input('Bill Amount2', min_value=-1000000, max_value=1000000, step=50000)
    #     bill_amt_3 = st.number_input('Bill Amount3', min_value=-1000000, max_value=1000000, step=50000)
    #     bill_amt_4 = st.number_input('Bill Amount4', min_value=-1000000, max_value=1000000, step=50000)
    #     bill_amt_5 = st.number_input('Bill Amount5', min_value=-1000000, max_value=1000000, step=50000)
    #     bill_amt_6 = st.number_input('Bill Amount6', min_value=-1000000, max_value=1000000, step=50000)
    #     st.markdown('---')
    #     pay_amt_1 = st.number_input('Pay Amount1', min_value=-1000000, max_value=1000000, step=50000)
    #     pay_amt_2 = st.number_input('Pay Amount2', min_value=-1000000, max_value=1000000, step=50000)
    #     pay_amt_3 = st.number_input('Pay Amount3', min_value=-1000000, max_value=1000000, step=50000)
    #     pay_amt_4 = st.number_input('Pay Amount4', min_value=-1000000, max_value=1000000, step=50000)
    #     pay_amt_5 = st.number_input('Pay Amount5', min_value=-1000000, max_value=1000000, step=50000)
    #     pay_amt_6 = st.number_input('Pay Amount6', min_value=-1000000, max_value=1000000, step=50000)

    #     submitted = st.form_submit_button('Predict')

    #     data_inf = {'limit_balance': limit_balance,
    #     'age_cat': age_cat,
    #     'sex': sex,
    #     'education_level': education_level,
    #     'marital_status': marital_status,
    #     'pay_1': int(pay_1),
    #     'pay_2': int(pay_2),
    #     'pay_3': int(pay_3),
    #     'pay_4': int(pay_4),
    #     'pay_5': int(pay_5),
    #     'pay_6': int(pay_6),
    #     'bill_amt_1': bill_amt_1,
    #     'bill_amt_2': bill_amt_2,
    #     'bill_amt_3': bill_amt_3,
    #     'bill_amt_4': bill_amt_4,
    #     'bill_amt_5': bill_amt_5,
    #     'bill_amt_6': bill_amt_6,
    #     'pay_amt_1': pay_amt_1,
    #     'pay_amt_2': pay_amt_2,
    #     'pay_amt_3': pay_amt_3,
    #     'pay_amt_4': pay_amt_4,
    #     'pay_amt_5': pay_amt_5,
    #     'pay_amt_6': pay_amt_6}

    # data_inf = pd.DataFrame([data_inf])
    # st.dataframe(data_inf)

    # if submitted:
    #     # Split between num column and cat column
    #     data_inf_num = data_inf[list_num_column_skew]
    #     data_inf_cat_o = data_inf[list_cat_column_ordinal]
    #     data_inf_cat_n = data_inf[list_cat_column_nominal]

    #     #scaling data
    #     data_inf_num_scaled = model_scaler_mm.transform(data_inf_num)
    #     #encoding data
    #     data_inf_cat_n_encoded = model_oh_encoder.transform(data_inf_cat_n).toarray()

    #     data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_o, data_inf_cat_n_encoded], axis=1)

    #     threshold = 0.6
    #     y_pred_new_proba_train = model_knn.predict_proba(data_inf_final)
    #     y_pred_new_train = np.where(y_pred_new_proba_train[:,1] >= threshold, 1, 0)

    #     st.write(f'# Predict Result : {str(int(y_pred_new_train))}')

if __name__ == '__main__':
    run()