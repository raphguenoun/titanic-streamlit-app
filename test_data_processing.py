"""
Tests unitaires pour le module data_processing
"""

import unittest
import pandas as pd
import numpy as np
from data_processing import (
    load_titanic_data, 
    clean_data, 
    encode_features, 
    prepare_features,
    preprocess_pipeline,
    create_prediction_input,
    get_feature_names
)


class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Configuration pour les tests"""
        # Créer un petit dataset de test
        self.test_data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 0, 1],
            'Pclass': [3, 1, 3, 1, 2],
            'Sex': ['male', 'female', 'female', 'male', 'female'],
            'Age': [22.0, 38.0, np.nan, 35.0, np.nan],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Fare': [7.25, 71.28, 7.92, 53.10, 8.05],
            'Embarked': ['S', 'C', 'S', 'S', np.nan]
        })
    
    def test_load_titanic_data_simulated(self):
        """Test du chargement des données simulées"""
        df = load_titanic_data()
        
        # Vérifier que le DataFrame n'est pas vide
        self.assertFalse(df.empty)
        
        # Vérifier les colonnes requises
        required_columns = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Vérifier les types de données
        self.assertTrue(df['Survived'].dtype in [int, float])
        self.assertTrue(df['Age'].dtype in [int, float])
    
    def test_clean_data(self):
        """Test du nettoyage des données"""
        df_clean = clean_data(self.test_data)
        
        # Vérifier qu'il n'y a plus de NaN dans Age
        self.assertFalse(df_clean['Age'].isna().any())
        
        # Vérifier qu'il n'y a plus de NaN dans Embarked
        self.assertFalse(df_clean['Embarked'].isna().any())
        
        # Vérifier que les valeurs de remplacement sont correctes
        self.assertEqual(df_clean.loc[df_clean['PassengerId'] == 5, 'Embarked'].iloc[0], 'S')
    
    def test_encode_features(self):
        """Test de l'encodage des features"""
        df_clean = clean_data(self.test_data)
        df_encoded = encode_features(df_clean)
        
        # Vérifier que les colonnes one-hot encodées sont présentes
        expected_columns = ['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                           'Pclass_1', 'Pclass_2', 'Pclass_3']
        
        for col in expected_columns:
            self.assertIn(col, df_encoded.columns)
        
        # Vérifier que les colonnes originales ont été supprimées
        self.assertNotIn('Sex', df_encoded.columns)
        self.assertNotIn('Embarked', df_encoded.columns)
        self.assertNotIn('Pclass', df_encoded.columns)
    
    def test_prepare_features(self):
        """Test de la préparation des features"""
        df_processed = preprocess_pipeline(self.test_data)[0]  # On récupère juste X
        df_with_target = self.test_data.copy()
        df_encoded = encode_features(clean_data(df_with_target))
        
        X, y = prepare_features(df_encoded)
        
        # Vérifier que X ne contient pas les colonnes exclues
        self.assertNotIn('PassengerId', X.columns)
        self.assertNotIn('Survived', X.columns)
        
        # Vérifier que y est correct
        self.assertEqual(len(y), len(self.test_data))
        self.assertEqual(y.name, 'Survived')
    
    def test_preprocess_pipeline(self):
        """Test du pipeline complet"""
        X, y = preprocess_pipeline(self.test_data)
        
        # Vérifier les dimensions
        self.assertEqual(len(X), len(self.test_data))
        self.assertEqual(len(y), len(self.test_data))
        
        # Vérifier qu'il n'y a pas de NaN
        self.assertFalse(X.isna().any().any())
        self.assertFalse(y.isna().any())
    
    def test_get_feature_names(self):
        """Test de la fonction get_feature_names"""
        feature_names = get_feature_names()
        
        # Vérifier le nombre de features
        self.assertEqual(len(feature_names), 12)
        
        # Vérifier que les features importantes sont présentes
        important_features = ['Age', 'Fare
