import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données (remplacez le chemin par celui de votre fichier CSV)
df = pd.read_csv('cosmetics.csv')

# Fonction pour filtrer les produits par type de peau et budget
def filter_products(df, skin_type, budget):
    if skin_type == 'dry':
        filtered_df = df[df['Dry'] == 1]
    elif skin_type == 'oily':
        filtered_df = df[df['Oily'] == 1]
    elif skin_type == 'combination':
        filtered_df = df[df['Combination'] == 1]
    elif skin_type == 'sensitive':
        filtered_df = df[df['Sensitive'] == 1]
    else:
        filtered_df = df

    if budget == 'economical':
        filtered_df = filtered_df[filtered_df['Price'] < 50]
    elif budget == 'standard':
        filtered_df = filtered_df[(filtered_df['Price'] >= 50) & (filtered_df['Price'] <= 150)]
    elif budget == 'premium':
        filtered_df = filtered_df[filtered_df['Price'] > 150]

    return filtered_df

# Vectorisation des descriptions des produits avec TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Ingredients'])

# Calcul de la similarité cosinus
def recommend_similar_products(product_name, tfidf_matrix, top_n=5):
    product_idx = df[df['Name'] == product_name].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[product_idx:product_idx+1], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    similar_products = df.iloc[similar_indices]
    return similar_products[['Name', 'Price', 'Brand', 'Rank', 'Ingredients']]

# Recommandation de routines complètes
def recommend_routine(filtered_df):
    cleansers = filtered_df[filtered_df['Label'] == 'Cleanser']
    serums = filtered_df[filtered_df['Label'] == 'Serum']
    moisturizers = filtered_df[filtered_df['Label'] == 'Moisturizer']
    
    routine = {
        'Cleanser': cleansers.iloc[0] if not cleansers.empty else None,
        'Serum': serums.iloc[0] if not serums.empty else None,
        'Moisturizer': moisturizers.iloc[0] if not moisturizers.empty else None,
    }
    return routine

# Interface utilisateur avec Streamlit
def main():
    st.title("BeautyMatch AI")
    st.sidebar.title("Préférences Utilisateur")

    # Collecter les préférences utilisateur
    skin_type = st.sidebar.selectbox("Quel est votre type de peau ?", ['dry', 'oily', 'combination', 'sensitive'])
    budget = st.sidebar.selectbox("Quel est votre budget ?", ['economical', 'standard', 'premium'])

    st.sidebar.markdown("### Résultats recommandés")

    # Filtrage des produits
    filtered_products = filter_products(df, skin_type, budget)

    # Affichage des produits filtrés
    st.write(f"### Produits recommandés pour une peau {skin_type} et un budget {budget}")
    st.write(filtered_products[['Name', 'Price', 'Brand', 'Rank']])

    # Recommandation de produits similaires
    if len(filtered_products) > 0:
        selected_product = st.selectbox("Choisissez un produit pour voir les similaires :", filtered_products['Name'])
        similar_products = recommend_similar_products(selected_product, tfidf_matrix)
        st.write(f"### Produits similaires à {selected_product}")
        st.write(similar_products)

        # Recommandation de routine complète
        routine = recommend_routine(filtered_products)
        st.write("### Routine complète recommandée")
        for category, product in routine.items():
            if product is not None:
                st.write(f"**{category}**: {product['Name']} - {product['Price']}€")
            else:
                st.write(f"**{category}**: Aucun produit trouvé")

    # Ajoutez une barre latérale interactive pour un design plus fluide et moderne
    st.sidebar.markdown("### Explorez nos produits!")
    st.sidebar.image("https://via.placeholder.com/150", caption="Cosmétiques", use_column_width=True)

# Lancer l'application Streamlit
if __name__ == '__main__':
    main()

