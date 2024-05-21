'''
Code for PriceLah, created based on reference code app.py provided by Farhan
'''
# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from string import capwords 


'''
Configuration and preparation
'''
# the csv contains the various features' values extracted from train.csv. First column is the 'street_name' as the key to lookup.  
data_file = r"./train_02_Engineered_filtering.csv"
df_data = pd.read_csv(data_file)

# trained model for price prediction
with open(r"./hdb_lr.pkl", 'rb') as rf:
    model = pickle.load(rf)

# StandardScalar object - to transform the data before calling to model (for prediction)
with open(r"./hdb_ss.pkl", 'rb') as rf_ss:
    ss = pickle.load(rf_ss)


## this function was not used at the end ## 
def predict_output(f_mid_storey,f_hdb_age, f_bus_interchange,f_mrt_interchange,f_Mall_Nearest_Distance):
    f_list = [f_mid_storey,f_hdb_age, f_bus_interchange,f_mrt_interchange,f_Mall_Nearest_Distance]
    df_feature = pd.DataFrame(f_list).transpose()
    return int(model.predict(df_feature))
    
# Define variables to hold the list for rendering the dropdown list on Streamlit page
town_list = ['ANG MO KIO','BEDOK','BISHAN','BUKIT BATOK','BUKIT MERAH','BUKIT PANJANG','BUKIT TIMAH',
             'CENTRAL AREA','CHOA CHU KANG','CLEMENTI','GEYLANG','HOUGANG','JURONG EAST','JURONG WEST',
             'KALLANG/WHAMPOA','MARINE PARADE','PASIR RIS','PUNGGOL','QUEENSTOWN','SEMBAWANG','SENGKANG',
             'SERANGOON','TAMPINES','TOA PAYOH','WOODLANDS','YISHUN']
flat_type_list =['1 ROOM|Improved','2 ROOM|2-room','2 ROOM|DBSS','2 ROOM|Improved','2 ROOM|Model A',
                 '2 ROOM|Premium Apartment','2 ROOM|Standard','3 ROOM|DBSS','3 ROOM|Improved',
                 '3 ROOM|Model A','3 ROOM|New Generation','3 ROOM|Premium Apartment','3 ROOM|Simplified',
                 '3 ROOM|Standard','3 ROOM|Terrace','4 ROOM|Adjoined flat','4 ROOM|DBSS','4 ROOM|Improved',
                 '4 ROOM|Model A','4 ROOM|Model A2','4 ROOM|New Generation','4 ROOM|Premium Apartment',
                 '4 ROOM|Premium Apartment Loft','4 ROOM|Simplified','4 ROOM|Standard','4 ROOM|Terrace',
                 '4 ROOM|Type S1','5 ROOM|Adjoined flat','5 ROOM|DBSS','5 ROOM|Improved','5 ROOM|Improved-Maisonette',
                 '5 ROOM|Model A','5 ROOM|Model A-Maisonette','5 ROOM|Premium Apartment','5 ROOM|Premium Apartment Loft',
                 '5 ROOM|Standard','5 ROOM|Type S2','EXECUTIVE|Adjoined flat','EXECUTIVE|Apartment',
                 'EXECUTIVE|Maisonette','EXECUTIVE|Premium Apartment','EXECUTIVE|Premium Maisonette',
                 'MULTI-GENERATION|Multi Generation']

# Define Dictionary for the lookup of certain values that Features need the value , but user is not inputing them 
flat_floor_area_dict = {
'1 ROOM|Improved':'333.684','2 ROOM|2-room':'546.273','2 ROOM|DBSS':'538.2','2 ROOM|Improved':'488.842618','2 ROOM|Model A':'499.2806387',
'2 ROOM|Premium Apartment':'557.4006486','2 ROOM|Standard':'483.8363636','3 ROOM|DBSS':'707.7915','3 ROOM|Improved':'701.8674609',
'3 ROOM|Model A':'771.765049','3 ROOM|New Generation':'751.8850872','3 ROOM|Premium Apartment':'726.298964','3 ROOM|Simplified':'692.9227399',
'3 ROOM|Standard':'650.9586619','3 ROOM|Terrace':'1172.024372','4 ROOM|Adjoined flat':'1255.8','4 ROOM|DBSS':'967.8689404','4 ROOM|Improved':'916.8072364',
'4 ROOM|Model A':'1067.037962','4 ROOM|Model A2':'960.3219419','4 ROOM|New Generation':'1002.752762','4 ROOM|Premium Apartment':'1020.13855',
'4 ROOM|Premium Apartment Loft':'1054.274','4 ROOM|Simplified':'909.5002162','4 ROOM|Standard':'833.34888','4 ROOM|Terrace':'1233.076',
'4 ROOM|Type S1':'1015.730182','5 ROOM|Adjoined flat':'1457.664','5 ROOM|DBSS':'1204.440815','5 ROOM|Improved':'1268.030378',
'5 ROOM|Improved-Maisonette':'1461.637895','5 ROOM|Model A':'1446.740383','5 ROOM|Model A-Maisonette':'1517.066656','5 ROOM|Premium Apartment':'1224.036963',
'5 ROOM|Premium Apartment Loft':'1593.072','5 ROOM|Standard':'1280.003039','5 ROOM|Type S2':'1142.340094','EXECUTIVE|Adjoined flat':'1740.595453',
'EXECUTIVE|Apartment':'1547.133797','EXECUTIVE|Maisonette':'1591.999053','EXECUTIVE|Premium Apartment':'1433.742297','EXECUTIVE|Premium Maisonette':'1727.2632',
'MULTI-GENERATION|Multi Generation':'1735.695'
}

max_floor_dict = {
'ANG MO KIO':'30','BEDOK':'25','BISHAN':'40','BUKIT BATOK':'30','BUKIT MERAH':'48','BUKIT PANJANG':'34','BUKIT TIMAH':'23','CENTRAL AREA':'50',
'CHOA CHU KANG':'25','CLEMENTI':'40','GEYLANG':'25','HOUGANG':'20','JURONG EAST':'40','JURONG WEST':'25','KALLANG/WHAMPOA':'40','MARINE PARADE':'25',
'PASIR RIS':'18','PUNGGOL':'19','QUEENSTOWN':'47','SEMBAWANG':'23','SENGKANG':'27','SERANGOON':'16','TAMPINES':'17','TOA PAYOH':'42','WOODLANDS':'33','YISHUN':'16'
}

region_dict = {'BUKIT BATOK':'west', 'BUKIT PANJANG':'west', 'JURONG WEST':'west', 'CHOA CHU KANG':'west', 'CLEMENTI':'west', 'JURONG EAST':'west', 
          'SEMBAWANG':'north', 'WOODLANDS':'north', 'YISHUN':'north', 
          'HOUGANG':'north_east', 'SENGKANG':'north_east', 'SERANGOON':'north_east', 'PUNGGOL':'north_east', 'ANG MO KIO':'north_east', 
          'BEDOK':'east', 'TAMPINES':'east', 'PASIR RIS':'east', 
          'KALLANG/WHAMPOA':'central', 'BISHAN':'central', 'GEYLANG':'central', 'BUKIT MERAH':'central', 'TOA PAYOH':'central', 'CENTRAL AREA':'central', 'QUEENSTOWN':'central', 'BUKIT TIMAH':'central', 'MARINE PARADE':'central'}

mrt_colour_dict = {'Kallang': 'green', 'Bishan':['orange','red'], 'Bukit Batok': 'red', 'Khatib': 'red', 'MacPherson': ['orange', 'blue'],
       'Kovan': 'purple', 'Bedok North': 'blue', 'Marymount':'orange', 'Sengkang': 'purple', 'Buangkok':'purple', 'Tampines': ['green', 'blue'],
       'Tiong Bahru':'green', 'Bukit Panjang':'blue', 'Marsiling':'red', 'Woodlands South':'brown', 'Admiralty':'red', 'Pioneer':'green', 'Braddell':'red',
       'Lakeside': 'green', 'Choa Chu Kang': 'red', 'Sembawang': 'red', 'Toa Payoh':'red','Geylang Bahru': 'blue',
       'Yew Tee':'red', 'Ang Mo Kio':'red', 'Telok Blangah':'orange', 'Tampines East':'blue',
       'Potong Pasir':'purple', 'Tampines West':'blue', 'Eunos':'green', 'Yio Chu Kang':'red',
       'Farrer Park':'purple', 'Bukit Gombak':'red', 'Clementi':'green', 'Yishun':'red', 'Punggol':'purple',
       'Jurong East': ['green','red'], 'Tanah Merah':'green', 'Chinese Garden':'green', 'Kembangan':'green',
       'Pasir Ris':'green', 'Jalan Besar':'blue', 'Hougang':'purple', 'Buona Vista':['green','orange'], 'Kaki Bukit':'blue',
       'Cashew':'blue', 'Bedok':'green', 'Boon Keng':'purple', 'Woodlands':'brown', 'Simei':'green', 'Boon Lay':'green',
       'Dakota':'orange', 'Redhill':'green', 'Canberra':'blue', 'Beauty World':'blue', 'Commonwealth':'green',
       'Lorong Chuan':'orange', 'Tai Seng':'orange', 'Bedok Reservoir':'blue', 'Holland Village':'orange',
       'Ubi':'blue', 'HarbourFront':['orange','purple'], 'Dover':'green', 'Chinatown':['blue','purple'], 'Queenstown':'green',
       'Mattar':'blue', 'one-north':'orange', 'Mountbatten':'orange', 'Serangoon':['orange','purple'], 'Farrer Road':'orange',
       'Lavender':'green', 'Outram Park':['green', 'purple'], 'Caldecott':['orange','brown'], 'Aljunied':'green', 'Little India':['blue', 'purple'],
       'Upper Changi':'blue', 'Bartley':'orange', 'Woodlands North':'brown', 'Paya Lebar':['green', 'orange'],
       'Tanjong Pagar':'green', 'Woodleigh':'purple', 'Hillview':'blue', 'Bencoolen':'blue',
       'Labrador Park':'orange', 'Rochor':'blue', 'Nicoll Highway':'orange', 'Clarke Quay':'purple',
       'Tan Kah Kee':'blue', 'Bras Basah':'orange', 'Changi Airport':['green','blue'], 'Bugis':['green','blue'],
       'Bendemeer':'blue', 'Botanic Gardens':['orange', 'blue'], 'Novena':['blue','red'] }

# Feature columns - for one-hot-encoding to match the model input 
feature_cols = ['mid_storey','hdb_age','bus_interchange','max_floor_lvl','Mall_Nearest_Distance','Hawker_Nearest_Distance','floor_area_sqft',
                'Ammenities_same_block','Ammenities_within_500m','Ammenities_within_1km','Ammenities_within_2km',
                'pri_sch_dist_vacancy','sec_sch_dist_cutoff','Tranc_YearMonth','mrt_no','mrt_nearest_distance',
                'town_BEDOK','town_BISHAN','town_BUKIT BATOK','town_BUKIT MERAH','town_BUKIT PANJANG','town_BUKIT TIMAH','town_CENTRAL AREA',
                'town_CHOA CHU KANG','town_CLEMENTI','town_GEYLANG','town_HOUGANG','town_JURONG EAST','town_JURONG WEST','town_KALLANG/WHAMPOA',
                'town_MARINE PARADE','town_PASIR RIS','town_PUNGGOL','town_QUEENSTOWN','town_SEMBAWANG','town_SENGKANG','town_SERANGOON',
                'town_TAMPINES','town_TOA PAYOH','town_WOODLANDS','town_YISHUN',
                'flat_type_2 ROOM','flat_type_3 ROOM','flat_type_4 ROOM','flat_type_5 ROOM','flat_type_EXECUTIVE','flat_type_MULTI-GENERATION',
                'region_east','region_north','region_north_east','region_west']


'''
Streamlit page rendering start from here onwards 
'''
# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='HDB Resale Price Predictor', 
                   page_icon='house-buildings', 
                   layout='centered', 
                   initial_sidebar_state='collapsed',
                   menu_items= {
                       'Get Help':'http://localhost:8501',
                       'Report a bug':'http://localhost:8501',
                       'About':'http://localhost:8501'                                              
                   })
st.subheader("Today's price, Tomorrow's home")

# Set title of the app
st.title('🏘️ PriceLah 🏘️ ')

town = st.selectbox("Town",town_list)  # Dropdown on streamlit screen
region = region_dict[town]

street = st.selectbox("Street Name",df_data[df_data['town']==town]['street_name'].value_counts(0).index ) # Dropdown on streamlit screen
max_floor = df_data[df_data['street_name']==street]['max_floor_lvl'].mean()
mrt_no = df_data[df_data['street_name']==street]['mrt_no'].values[0]
mrt_distance = df_data[df_data['street_name']==street]['mrt_nearest_distance'].mean()
bus_interchange = df_data[df_data['street_name']==street]['bus_interchange'].values[0]
mall_distance =df_data[df_data['street_name']==street]['Mall_Nearest_Distance'].mean()
hawker_distance = df_data[df_data['street_name']==street]['Hawker_Nearest_Distance'].mean()
ammenities_same_block = df_data[df_data['street_name']==street]['Ammenities_same_block'].sort_values(ascending=False).values[0]
ammenities_500m = df_data[df_data['street_name']==street]['Ammenities_within_500m'].sort_values(ascending=False).values[0]
ammenities_1km = df_data[df_data['street_name']==street]['Ammenities_within_1km'].sort_values(ascending=False).values[0]
ammenities_2km = df_data[df_data['street_name']==street]['Ammenities_within_2km'].sort_values(ascending=False).values[0]
pri_school_dist_vacancy =  df_data[df_data['street_name']==street]['pri_sch_dist_vacancy'].mean()
sec_school_dist_cutoff = df_data[df_data['street_name']==street]['sec_sch_dist_cutoff'].mean()

flat_type_model = st.selectbox("Flat Model",flat_type_list) # Dropdown on streamlit screen
flat_type = flat_type_model.split('|')[0]
floor_area = float(flat_floor_area_dict[flat_type_model])

storey = st.slider('Storey', 1, 45, 6) # Slider on streamlit screen
hdb_age = st.slider('HDB Age',  6, 50, 10) # Slider on streamlit screen

button = st.button('Predict')  # "Predict" button on streamlit screen


# once "Predict" button on screen is clicked, then this if condition turn TRUE, and execute ... 
if button:

    #formulate this dictionary to hold all the input (from user on Streamlit page, or "derived" from the various lookup)
    user_input = {
        "Tranc_YearMonth": 202103,
        "mid_storey": storey, 
        "region": region,
        "hdb_age": hdb_age,
        "mrt_no" : mrt_no,
        "mrt_nearest_distance": mrt_distance,
        "Hawker_Nearest_Distance" : hawker_distance ,
        "bus_interchange":bus_interchange, 
        "Mall_Nearest_Distance": mall_distance,
        "town": town,
        "floor_area_sqft":floor_area,
        "flat_type": flat_type,
        "max_floor_lvl": max_floor,
        "Ammenities_same_block": ammenities_same_block ,
        "Ammenities_within_500m" : ammenities_500m ,
        "Ammenities_within_1km" : ammenities_1km ,
        "Ammenities_within_2km" : ammenities_2km ,
        "pri_sch_dist_vacancy" : pri_school_dist_vacancy,
        "sec_sch_dist_cutoff" : sec_school_dist_cutoff
    }

    df_input = pd.DataFrame([user_input])

    # #dummified the input (to prediction), to follow the columns in "feature_cols", with reindex, and any empty value to fill with 0 
    df_input = pd.get_dummies(df_input).reindex(columns=feature_cols, fill_value=0)

    result = int(model.predict(ss.transform(df_input)))
    st.subheader('Prediction ... ')
    st.success(f'The value of the house is ${result}')
    