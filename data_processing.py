"""
Module de traitement des données pour l'application Titanic
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


def load_titanic_data(filepath: str = None) -> pd.DataFrame:
    """
    Charge les données du Titanic depuis un fichier CSV ou génère des données simulées.
    
    Args:
        filepath (str): Chemin vers le fichier CSV (optionnel)
    
    Returns:
        pd.DataFrame: DataFrame contenant les données du Titanic
    """
    if filepath:
        # Charger depuis un fichier réel
        return pd.read_csv(filepath, dtype={'Survived': float, 'SibSp': float, 'Parch': float, 'Fare': float})
    else:
        # Générer des données simulées pour la démo
        np.random.seed(42)
        n_samples = 891
        
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29.7, 14.5, n_samples).clip(0, 80),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01]),
            'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.001, 0.001]),
            'Fare': np.random.exponential(15, n_samples).clip(0, 500),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        }
        
        df = pd.DataFrame(data)
        
        # Ajout de quelques valeurs manquantes pour simuler les vraies données
        missing_age_idx = np.random.choice(df.index, size=int(0.2 * n_samples), replace=False)
        df.loc[missing_age_idx, 'Age'] = np.nan
        
        missing_embarked_idx = np.random.choice(df.index, size=2, replace=False)
        df.loc[missing_embarked_idx, 'Embarked'] = np.nan
        
        return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en gérant les valeurs manquantes.
    
    Args:
        df (pd.DataFrame): DataFrame brut
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    df_clean = df.copy()
    
    # Remplacer les valeurs manquantes numériques par la moyenne
    numeric_columns = ['Age', 'Fare', 'SibSp', 'Parch']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Remplacer les valeurs manquantes catégorielles
    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'] = df_clean['Embarked'].fillna('S')  # Southampton est le plus fréquent
    
    return df_clean


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles en utilisant le one-hot encoding.
    
    Args:
        df (pd.DataFrame): DataFrame avec variables catégorielles
    
    Returns:
        pd.DataFrame: DataFrame avec variables encodées
    """
    df_encoded = df.copy()
    
    # Variables catégorielles à encoder
    categorical_cols = ['Sex', 'Embarked', 'Pclass']
    
    # One-hot encoding
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=col, drop_first=False)
    
    return df_encoded


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les features X et la target y pour l'entraînement.
    
    Args:
        df (pd.DataFrame): DataFrame preprocessé
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X et target y
    """
    # Colonnes à exclure des features
    exclude_cols = ['PassengerId', 'Survived']
    
    # Sélectionner les features
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_columns]
    
    # Target
    y = df['Survived'] if 'Survived' in df.columns else None
    
    return X, y


def preprocess_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline complet de preprocessing.
    
    Args:
        df (pd.DataFrame): DataFrame brut
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features X et target y preprocessées
    """
    # Étapes du preprocessing
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    X, y = prepare_features(df_encoded)
    
    return X, y


def get_feature_names() -> list:
    """
    Retourne la liste des noms de features après preprocessing.
    
    Returns:
        list: Liste des noms de features
    """
    return [
        'Age', 'SibSp', 'Parch', 'Fare',
        'Sex_female', 'Sex_male',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Pclass_1', 'Pclass_2', 'Pclass_3'
    ]


def create_prediction_input(age: float, sibsp: int, parch: int, fare: float, 
                          sex: str, embarked: str, pclass: int) -> pd.DataFrame:
    """
    Crée un DataFrame d'input pour la prédiction à partir des paramètres utilisateur.
    
    Args:
        age (float): Âge du passager
        sibsp (int): Nombre de frères/sœurs/époux(se)
        parch (int): Nombre de parents/enfants
        fare (float): Prix du ticket
        sex (str): Sexe ('male' ou 'female')
        embarked (str): Port d'embarquement ('S', 'C', 'Q')
        pclass (int): Classe (1, 2, 3)
    
    Returns:
        pd.DataFrame: DataFrame formaté pour la prédiction
    """
    # Créer le vecteur de features
    input_data = pd.DataFrame({
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_female': [1 if sex == 'female' else 0],
        'Sex_male': [1 if sex == 'male' else 0],
        'Embarked_C': [1 if embarked == 'C' else 0],
        'Embarked_Q': [1 if embarked == 'Q' else 0],
        'Embarked_S': [1 if embarked == 'S' else 0],
        'Pclass_1': [1 if pclass == 1 else 0],
        'Pclass_2': [1 if pclass == 2 else 0],
        'Pclass_3': [1 if pclass == 3 else 0]
    })
    
    # Réorganiser les colonnes dans l'ordre correct
    feature_order = get_feature_names()
    input_data = input_data[feature_order]
    
    return input_data
