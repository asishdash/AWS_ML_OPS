import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle as pkl


housing_dataset = fetch_openml(name="house_prices",as_frame=True)
X= housing_dataset.data[['GrLivArea','YearBuilt']]
Y = housing_dataset.target

df = housing_dataset.data
#print(df.columns )
#print(type(housing_dataset) )

df_new = pd.DataFrame(data=housing_dataset.data, columns=housing_dataset.feature_names)
#print(df_new.columns)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
with open("model.pkl",'wb') as f :
    pkl.dump(model,f)

st.title('Ames Housing Price Prediction')
GrLivArea = st.number_input('Above grade living area in square feet', value=1500)
YearBuilt = st.number_input('Original construction date', value=2000)
input_data = pd.DataFrame({'GrLivArea': [GrLivArea], 'YearBuilt': [YearBuilt]})
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Predicted House Price: ${prediction[0]:.2f}')


# st.title('Hello, Streamlit!')
# st.write('Welcome to your first Streamlit app.')

# to run >>  streamlit run << .py file path>
