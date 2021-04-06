import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.express as px
from datetime import datetime
from PIL import Image

st.set_page_config(layout='wide')

foto = Image.open('bike_rental.jpg')
st.image(foto,
         caption='Logo do Streamlit',
         use_column_width=False)

st.write("""
# Predicting demand to bike rentals App

**This application predicts the demand for bike rental per hour from a company specialized in this type of service** 

""")

st.sidebar.header('Select features to prediction rental bikes')



# Collects user input features into dataframe

def user_input_features():
    
    season = st.sidebar.selectbox('Season, 1:Winter, 2:Spring, 3:Summer, 4:Autumn',('1','2', '3','4'))
    yr = st.sidebar.selectbox('Select Year 0: 2011, 1: 2012',('0','1'))
    mnth = st.sidebar.selectbox('Select Month',('1','2','3', '4', '5', '6','7','8', '9', '10','11','12'))
    holiday = st.sidebar.selectbox('Holiday, 0: No, 1: Yes',('0','1'))
    weekday =  st.sidebar.selectbox('Weekday',('0','1','2','3','4','5','6'))
    workingday = st.sidebar.selectbox('Workingday 0: No, 1: Yes',('0','1'))
    weathersit = st.sidebar.selectbox('Weathersit',('1','2','3','4'))
    dteday = st.sidebar.slider('Choose One Day', 1, 730, 200)
    instant= st.sidebar.slider('Instant: registry', 1, 17300, 10000)
    hr = st.sidebar.slider('Hour', 1, 23, 10)
    temp = st.sidebar.slider('Normalizad Temperature Variation', 0.02, 1.00, 0.05)
    atemp = st.sidebar.slider('Temperature Variation', 0.0, 1.0, 0.5)
    hum = st.sidebar.slider('Normalized Humidity', 0.0, 1.0, 0.5)
    windspeed = st.sidebar.slider('Windspeed', 0.0, 0.85, 0.50)
    data = {'season': season,
                'yr': yr,
                'mnth': mnth,
                'holiday': holiday,
                'weekday': weekday,
                'workingday': workingday,
                'weathersit': weathersit,
                'dteday':dteday,
                'instant':instant,
                'hr': hr,
                'temp': temp,
                'atemp': atemp,
                'hum': hum,
                'windspeed': windspeed}
    features = pd.DataFrame(data, index=[0])
    return features
df0 = user_input_features()

# Defining the encoder and normalization commands
label = LabelEncoder()
stand = StandardScaler().fit(df0)

# Load dataset and split in features and target
bike = pd.read_csv('bike_app.csv')
X = bike.drop('cnt', axis= 1)
Y = bike['cnt']

st.header('Overview Dataset')
st.write(bike)
# Normalization dataset
df0 = stand.transform(df0)


st.subheader('User Input parameters')
st.write(df0)

# Label Encoder data
X['dteday'] = label.fit_transform(X['dteday'])

# Rescaler data
stand = StandardScaler().fit(X)
X = stand.transform(X)

# Load  pickle with RamdonFlorestClassofoier
load_clf = pickle.load(open('model_best.pkl', 'rb'))

load_clf.fit(X, Y)

# Predition 
prediction = load_clf.predict(df0)
prediction_proba = load_clf.score(X, Y)

st.subheader('Prediction of the number of bikes rented')
st.write(prediction)

st.subheader('Accuracy of Prediction ')
st.write("{:.2f} %".format(100*prediction_proba))

# Show plot rental bikes with date
st.header('Variation of bike rentals depending on the date')
bike['dteday'] = pd.to_datetime(bike['dteday']).dt.strftime('%Y-%m-%d')

min_date = datetime.strptime( bike['dteday'].min(), '%Y-%m-%d' )
max_date = datetime.strptime(bike['dteday'].max(), '%Y-%m-%d')

st.sidebar.subheader('Show rental bikes with date')
f_date = st.sidebar.slider(' Select Date', min_date, max_date, min_date)

bike['dteday'] = pd.to_datetime(bike['dteday'])

df =  bike.loc[bike['dteday'] < f_date]
df = df[['dteday', 'cnt']].groupby('dteday').mean().reset_index()

fig = px.line(df, x = 'dteday', y = 'cnt')
st.plotly_chart(fig, use_container_width=True)


st.header("Rental quantity distribution")

# Show plot rental bikes distribution
st.sidebar.subheader('Show rental bikes distribution')
st.sidebar.write('Select Date')
price_min = int(bike['cnt'].min())
price_max = int(bike['cnt'].max())
price_avg = int(bike['cnt'].mean())

f_price = st.sidebar.slider('cnt', price_min, price_max, price_avg)
df = bike.loc[bike['cnt']< f_price]


fig = px.histogram(df, x = 'cnt', nbins=50, color_discrete_sequence=['green'])
st.plotly_chart(fig, use_container_width= True)

st.header('Varacional rental bikes')


st.sidebar.subheader('Show rental bikes with some features')
# Show plot rental bikes with temperature
st.sidebar.write('Select temperature')
min_temp = 0.0
max_temp = 1.0
avg_temp = 0.5

c1, c2 = st.beta_columns((1, 1))
c1.subheader('cnt X temperature')
f_temp = st.sidebar.slider('cnt variation as a function of temperature variation', min_temp, max_temp, avg_temp)

df =  bike.loc[bike['temp'] < f_temp]
df = df[['temp', 'cnt']].groupby('temp').mean().reset_index()

fig = px.line(df, x = 'temp', y = 'cnt')
c1.plotly_chart(fig, use_container_width=True)

st.sidebar.write('Select windspeed')
# Show plot rental bikes  with wind speed
win_min = 0.0
win_max = 0.8507
win_avg = 0.3

f_wind = st.sidebar.slider('cnt variation as a function of wind speed', min_temp, max_temp, avg_temp)

df =  bike.loc[bike['windspeed'] < f_wind]
df = df[['windspeed', 'cnt']].groupby('windspeed').mean().reset_index()

c2.subheader('cnt X wind speed')
fig = px.line(df, x = 'windspeed', y = 'cnt')
c2.plotly_chart(fig, use_container_width=True)