import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Survie - Titanic",
    page_icon="🚢",
    layout="wide"
)

# Titre principal
st.title("🚢 Prédiction de Survie du Titanic")
st.markdown("*Application de Machine Learning pour prédire la survie des passagers*")

# Fonction pour créer des données simples
@st.cache_data
def load_real_data():
    df = pd.read_csv('titanic_data.csv')
    
    # Sélectionner UNIQUEMENT les colonnes numériques nécessaires
    columns_to_keep = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    # Filtrer pour ne garder que les colonnes qui existent ET qui nous intéressent
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    
    # Conversion explicite des colonnes numériques
    for col in ['Age', 'SibSp', 'Parch', 'Fare']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    
    # Nettoyage des catégorielles
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna('S')
    
    return df
    
    # Données plus simples sans erreurs de probabilité
    data = {
        'Age': np.random.uniform(1, 80, n),
        'Fare': np.random.uniform(5, 100, n),
        'Pclass': np.random.choice([1, 2, 3], n),
        'Sex': np.random.choice(['male', 'female'], n),
        'SibSp': np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
        'Parch': np.random.choice([0, 1, 2], n, p=[0.8, 0.15, 0.05]),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Créer la variable Survived basée sur des règles logiques
    survived = []
    for _, row in df.iterrows():
        prob = 0.5  # probabilité de base
        if row['Sex'] == 'female':
            prob += 0.3
        if row['Pclass'] == 1:
            prob += 0.2
        elif row['Pclass'] == 2:
            prob += 0.1
        if row['Age'] < 16:
            prob += 0.2
        
        survived.append(1 if np.random.random() < prob else 0)
    
    df['Survived'] = survived
    return df

# Fonction de preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # One-hot encoding pour les variables catégorielles
    df_processed = pd.get_dummies(df_processed, columns=['Sex', 'Embarked', 'Pclass'])
    
    return df_processed

# Charger les données
df = load_real_data()
df_processed = preprocess_data(df)

# Préparer X et y
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner les modèles
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Interface utilisateur
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisir une page", ["🎯 Prédiction", "📊 Données", "🏆 Performance"])

if page == "🎯 Prédiction":
    st.header("🎯 Faire une Prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations Personnelles")
        age = st.slider("Âge", 1, 80, 30)
        sex = st.selectbox("Sexe", ["male", "female"])
        
    with col2:
        st.subheader("Détails du Voyage")
        pclass = st.selectbox("Classe", [1, 2, 3])
        fare = st.slider("Prix du ticket", 5.0, 100.0, 32.0)
    
    sibsp = st.selectbox("Frères/sœurs/époux", [0, 1, 2])
    parch = st.selectbox("Parents/enfants", [0, 1, 2])
    embarked = st.selectbox("Port", ["S", "C", "Q"])
    
    model_choice = st.selectbox("Modèle", ["Decision Tree", "Random Forest"])
    
    if st.button("🔮 Prédire la Survie", type="primary"):
        # Créer input pour prédiction
        input_data = pd.DataFrame({
            'Age': [age],
            'Fare': [fare],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Sex_female': [1 if sex == 'female' else 0],
            'Sex_male': [1 if sex == 'male' else 0],
            'Embarked_C': [1 if embarked == 'C' else 0],
            'Embarked_Q': [1 if embarked == 'Q' else 0],
            'Embarked_S': [1 if embarked == 'S' else 0],
            'Pclass_1': [1 if pclass == 1 else 0],
            'Pclass_2': [1 if pclass == 2 else 0],
            'Pclass_3': [1 if pclass == 3 else 0]
        })
        
        # Réorganiser selon l'ordre d'entraînement
        input_data = input_data[X.columns]
        
        # Prédiction
        if model_choice == "Decision Tree":
            prediction = dt_model.predict(input_data)[0]
            probability = dt_model.predict_proba(input_data)[0]
        else:
            prediction = rf_model.predict(input_data)[0]
            probability = rf_model.predict_proba(input_data)[0]
        
        # Affichage
        if prediction == 1:
            st.success("🎉 AURAIT SURVÉCU !")
            st.balloons()
        else:
            st.error("💔 N'AURAIT PAS SURVÉCU")
        
        # Graphique des probabilités
        prob_df = pd.DataFrame({
            'Résultat': ['Décès', 'Survie'],
            'Probabilité': probability
        })
        
        fig = px.bar(prob_df, x='Résultat', y='Probabilité', color='Probabilité')
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Données":
    st.header("📊 Analyse des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Taux de survie
        survival_rate = df['Survived'].mean()
        st.metric("Taux de survie global", f"{survival_rate:.1%}")
        
        # Distribution par sexe
        fig1 = px.histogram(df, x='Sex', color='Survived', 
                           title="Survie par Sexe", barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Distribution par classe
        fig2 = px.histogram(df, x='Pclass', color='Survived', 
                           title="Survie par Classe", barmode='group')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Age distribution
        fig3 = px.histogram(df, x='Age', title="Distribution des Âges")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Tableau de données
    st.subheader("Aperçu des Données")
    st.dataframe(df.head(10))

else:  # Performance
    st.header("🏆 Performance des Modèles")
    
    # Prédictions
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decision Tree")
        dt_acc = accuracy_score(y_test, dt_pred)
        st.metric("Précision", f"{dt_acc:.2%}")
        
    with col2:
        st.subheader("Random Forest") 
        rf_acc = accuracy_score(y_test, rf_pred)
        st.metric("Précision", f"{rf_acc:.2%}")
    
    # Comparaison
    comparison = pd.DataFrame({
        'Modèle': ['Decision Tree', 'Random Forest'],
        'Précision': [dt_acc, rf_acc]
    })
    
    fig = px.bar(comparison, x='Modèle', y='Précision', 
                 title="Comparaison des Modèles")
    st.plotly_chart(fig, use_container_width=True)
    
    # Importance des features (Random Forest)
    if hasattr(rf_model, 'feature_importances_'):
        st.subheader("Importance des Variables")
        importance_df = pd.DataFrame({
            'Variable': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(importance_df.head(8), 
                        x='Importance', y='Variable', 
                        orientation='h',
                        title="Variables les Plus Importantes")
        st.plotly_chart(fig_imp, use_container_width=True)

# Informations
st.sidebar.markdown("---")
st.sidebar.markdown("**Projet Titanic ML**")
st.sidebar.markdown("Application Streamlit de prédiction de survie")
