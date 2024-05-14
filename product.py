import pandas as pd
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_concatenate_csv(directory_path):
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except pd.errors.EmptyDataError:
                print(f"EmptynDataFrame from file: {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")


    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()   ## Return an empty DataFrame if not data is found
    
# Load and Concatenate the data
directory = '/Users/jerroldl/Downloads/projects/finalproject/item_list'
product_data = load_and_concatenate_csv(directory)

#Cheak the DataFrame is empty

if product_data.empty:
    raise ValueError("The DataFrame is empty. Please cheak your CSV file.")

if 'name' not in product_data.columns:
    raise ValueError("The 'name' column is missing from the dataset.")

#Fill any missing vslue in the 'name' column

product_data['name'] = product_data['name'].fillna('Unknown Name')

if product_data is None or 'name' not in product_data.columns:
    raise Exception("Data loading error or 'name' column missing.")
if product_data is None:
    raise Exception("Failed to load or data is empty.")

def preprocess_data(product_data):
    #Handle missing values
    product_data['name'] = product_data['name'].fillna('')
    
    #Removing duplicates entries based on the 'name' column
    product_data = product_data.drop_duplicates(subset = 'name')

    #reating a comopsite feagure
    product_data['composite_text'] = product_data['name'] + ' ' + product_data['main_category'] + ' ' + product_data['sub_category']

    return product_data

#Preprocess the data
def preprocess_data(product_data):
    product_data = product_data.copy()
    product_data['name'] = product_data['name'].fillna('')
    product_data = product_data.drop_duplicates(subset = 'name')
    product_data['composite_feature'] = product_data['name'] + ' ' + product_data['main_category'] + ' ' + product_data['sub_category']
    return product_data

product_data = preprocess_data(product_data)

if 'composite_feature' not in product_data.columns:
    raise ValueError("The 'composite_feature' column is missing from the Dataset after preprocessing")


tfidf_vector = TfidfVectorizer(stop_words='english')
tfidf_metrix = tfidf_vector.fit_transform(product_data['composite_feature'])

# Function to get recommendations
def get_recommendation(selected_product, tfidf_metrix, num_recommendations=5):
    #Find the index of the product that matches the selected product
    idx_list = product_data.index[product_data['name']== selected_product].tolist()

    #Cheak if the index list is empty
    if not idx_list:
        #Handle the scenario where the product isn't found
        return pd.DataFrame(columns = ['name', 'image'])
    idx = idx_list[0]

    # Check if the index is within the range of the TF-IDF metrix
    if idx < 0 or idx >= tfidf_metrix.shape[0]:
        raise ValueError(f"Index {idx} is out of range for TF-IDF matrix with shape {tfidf_metrix.shape}")
    
    # Check if the TF-IDF metrix contains at least one sample
    #if tfidf_metrix.shape[0] == 0:
        #raise ValueError("TF-IDF matrix is empty. There are no samples to compute cosine similarity.")


    #Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_metrix[idx:idx+1], tfidf_metrix)

    #Get pairs of (product index, similarity score)
    sim_score = list(enumerate(cosine_sim[0]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)

    #Get the indicates of the most similar products
    sim_indices = [i[0] for i in sim_score[1:num_recommendations+1]]

    #Cheak if the indices are in DataFrame index
    valid_indices = [i for i in sim_indices if i in product_data.index]

    #Return the most similar products
    return product_data.loc[valid_indices, ['name', 'image']]

#Streamlit UI
st.title('Product Recommendiation System')

#Display categories
categories = product_data['main_category'].unique()
selected_category = st.selectbox('Select a Category', categories, key='category_select')

#product selection based on selected category
filtered_products = product_data[product_data['main_category']== selected_category]
if not filtered_products.empty:
    selected_product = st.selectbox('Select a product', filtered_products['name'].unique(), key='product_select')
else:
    st.write("No product available in this category.")

#Get Recommendations
    
if st.button('Get Recommensations'):
    recommendations = get_recommendation(selected_product, tfidf_metrix)
    if not recommendations.empty:
        st.subheader('Recommendiation Products:')
        for _, row in recommendations.iterrows():
            st.write(row['name'])
            st.image(row['image'], width=200)

#User feedback
feedback = st.text_input("Feedback on Recommendation (Optional)")
if st.button("Submit Feedback"):
    st.write("Thankyou for your feedback!")