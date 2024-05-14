import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

#Function to load the data (Replace with actual code data)
@st.cache_data

def load_data():
    data = pd.read_csv("/Users/jerroldl/Downloads/classification_data.csv")
    return data

## Function to create a Distribution Data

def dist_plot(data, column, title):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot()

def train_model(X_train, Y_train):
    model = RandomForestClassifier()
    ## Train the model using the training data

    model.fit(X_train, Y_train)
    return model

##Streamlit application layout

st.title('Final Project Model-building, OCR, product-recommendation, NLP')

#Creating a sidebar 

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "EDA", "Model Building", "Final Output"],
                           icons=['house', 'bar-chart-line','wrench', 'file-text'],default_index=0)
    
#Home Page
    
if selected == "Home":
    st.subheader("About the Dataset")
    data = load_data()

    #Display the basic information about dataset
    if st.checkbox('Show Dataset'):
        st.dataframe(data)

    st.write('Data Dimension:', data.shape)
    st.write('column Name:', data.columns.tolist())
    st.write('Data Type', data.dtypes)

    if st.checkbox('Show Summery'):
        st.dataframe(data.describe())

    sns.set(style="whitegrid")
    st.set_option('deprecation.showPyplotGlobalUse',False)
#Ploating the distribution of the target variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x='has_converted', data=data)
    plt.title('Distribution of target variable(has_converted)')
    plt.xlabel('Has Converted')
    plt.ylabel('Count')
    st.pyplot()


#EDA page
    
elif selected == "EDA":
    st.subheader("Exploratory Data Analysis")
    data = load_data()

    #Ensure the columns are exist in your dataset
    if all(col in data.columns for col in ['count_session', 'count_hit', 'transactionRevenue']):
        dist_plot(data, 'count_session', 'Distribution of session Counts')
        dist_plot(data, 'count_hit', 'Distributio of Hit Counts')
        dist_plot(data, 'transactionRevenue', 'Distribution of Transaction Revenue')
    else:
        st.error("Required columns are no in the dataset")
    ## Selecting numeric columns for correlation matrix
        
    numeric_data = data.select_dtypes(include=['int64', 'float64'])

#Plotting the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".1f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot()

    st.subheader('Numerical Data Distribution')
    num_col = data.select_dtypes(include=['int64', 'float64']).columns
    for col in num_col:
        st.subheader(f'Histogram for {col}')
        fig, ax = plt.subplots()
        sns.histplot(data[col], ax=ax)
        st.pyplot(fig)
    
    st.subheader('Target variable Distribution')
    fig, ax = plt.subplots(figsize=(8, 8))
    data['has_converted'].value_counts().plot(kind='pie', autopct = '%1.1f%%', ax=ax)
    st.pyplot(fig)

elif selected == "Final Output":
    st.subheader("Final Output")
    data = load_data()
    df = data
    def display_data(df):
        #Total clik and counts
        total_clicks = df['count_hit'].sum()
        total_selection = df['count_session'].sum()

        #calculating device count
        dev_counts = df['device_deviceCategory'].value_counts()

        #calculating coversion counts
        conversion = df['has_converted'].value_counts()

        return total_clicks, total_selection, dev_counts, conversion
    #Streamlit app layout
    st.title('Data Analysis App')
    data = load_data()
    #Button to update counts
    if st.button('Update Counts'):
        total_clicks, total_selection, dev_counts, conversion = display_data(data)
        st.write(f'Total click count:{total_clicks}')
        st.write(f'Total session count:{total_selection}')
        st.write(f'Device Category Breakdown:')
        st.write(dev_counts)
        st.write('conversion count:')
        st.write(f'conevrted:{conversion.get(1, 0)}')
        st.write(f'Not converted:{conversion.get(0, 0)}')

elif selected == "Model Building":
    st.subheader("Model Building")
    data = load_data()

    data = data[['count_session', 'count_hit', 'channelGrouping', 'device_browser', 'geoNetwork_region', 'has_converted', 'transactionRevenue']].dropna()

    #Encoding Categorical variable and scaling 

    cat_feature = ['channelGrouping', 'device_browser', 'geoNetwork_region']
    num_feature = ['count_session', 'count_hit'] ## numerical feature

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_feature),
            ('cat', cat_transformer, cat_feature)])
    
    #Define target for regression and classification
    y_reg = data['transactionRevenue'] #for Regression
    y_cls = data['has_converted']  #for classification
    X = data.drop(['has_converted', 'transactionRevenue'],axis=1)


    # Spliting the data into training and testing sets

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)


    #Model Building
    # 1.Regression Model (Random Forest Regressor)
    regressor = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    
    #2. Classification Model (Random Forest Classifier)
    classifier = Pipeline(steps=[('preprossor', preprocessor),
                                 ('classifier',RandomForestClassifier(n_estimators=100, random_state=42))])
    
    #3. Clustering Model(KMeans)
    # For Clustering, we need to preprocess the entire dataset as there's no y target
    X_preprocessed = preprocessor.fit_transform(X)
    Kmeans = KMeans(n_clusters=3, random_state=42)

    #Fit the model
    regressor.fit(X_train,y_train_reg)
    classifier.fit(X_train, y_train_cls)
    Kmeans.fit(X_preprocessed)

    #Predictions for evaluation
    y_pred_reg = regressor.predict(X_test)
    y_pred_cls = classifier.predict(X_test)

    # For Clustering, assign each sample to a cluster
    clusters = Kmeans.predict(X_preprocessed)

    # Prepere result for plotting 
    results = {
        "Regression Model Predictions": y_pred_reg,
        "Classification Model Prediction": y_pred_cls,
        "Clustering Model Assignment": clusters
    }

    results

    def create_plots(y_test_reg, y_pred_reg, y_test_cls, y_pred_cls, clusters):
        plt.figure(figsize=(18,6))

        #Regression plot

        plt.subplot(1, 3, 1)
        sns.scatterplot(x=y_test_reg,y=y_pred_reg)
        plt.title('Regression Model: Actual vs Predicted Revenue')
        plt.xlabel('Actual Revenue')
        plt.ylabel('Predicted Revenue')

        #Classification plot
        plt.subplot(1, 3, 2)
        sns.scatterplot(x=y_test_cls,y=y_pred_cls)
        plt.title('Classification Model: Actual vs Predicted conversion')
        plt.xlabel('Actual conversion(0=No, 1=Yes)')
        plt.ylabel('Predicted Conversion Probability')

        #Clustrering plot
        plt.subplot(1, 3, 3)
        sns.histplot(clusters, kde=False, bins=3)
        plt.title('Clustering Model: Visitor Segments')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Visitors')

        return plt
    
    #Streamlit app

    st.title("Model Visulization")

    #Grnerate Plots
    fig = create_plots(y_test_reg, y_pred_reg, y_test_cls, y_pred_cls, clusters)

    #Display plots in streamlit
    st.pyplot(fig)

    def preprocess_data(data):
        #For numerical columns: fill missing value wit mean
        for col in data.select_dtypes(include='number').columns:
            data[col].fillna(data[col].mean(), inplace=True)

        #For categorical columns: fill missing values with mode
        for col in data.select_dtypes(include='object').columns:
            data[col].fillna(data[col].mode()[0], inplace=True)

        return data
    
##Assuming 'data' is your pandas DataFrame
    data = preprocess_data(data)

    regression_predictions = np.array([29676397., 0., 879667380., 28656201., 57425197., 0.])
    classification_predictions = np.array([1, 0, 1, 1, 0, 0])
    clustering_assignment = np.array([0, 0, 0, 2, 2, 2])

    #Streamlit application

    st.title("Model Predictions Visualization")

    # Regression Model Predictions

    st.subheader("Regression Model Predictions")
    st.write(pd.DataFrame(regression_predictions, columns = ['predicted value']))


    #plot for Regression Predictions

    plt.figure(figsize=(10, 4))
    sns.histplot(regression_predictions, kde=True)
    plt.title('Distribution of Regresion Predictions')
    st.pyplot(plt)

    #Classification Model Prediction

    st.subheader("Classification Model Predictions")
    st.write(pd.DataFrame(classification_predictions, columns=['Predicted Class']))

    #plot for classification predictions
    plt.figure(figsize=(10, 4))
    sns.countplot(x=classification_predictions)
    plt.title('Count of classification Predictions')
    st.pyplot(plt)

    #Clustering Model Assignments
    st.subheader("Clustering Model Asignment")
    st.write(pd.DataFrame(clustering_assignment, columns=['Cluster Assignment']))

    # Plot for clustering Assignment
    plt.figure(figsize=(10, 4))
    sns.countplot(x=clustering_assignment)
    plt.title('count of Clustering Assignment')
    st.pyplot(plt)