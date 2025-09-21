# 🚢 Titanic Survival Prediction App

Application Streamlit de Machine Learning pour prédire la survie des passagers du Titanic.

## 🚀 Installation et Lancement
```bash
# Cloner le repository
git clone https://github.com/VOTRE-USERNAME/titanic-streamlit-app.git
cd titanic-streamlit-app

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py

## Docker - TESTÉ ET FONCTIONNEL
```bash
# Construction de l'image (✅ réussie)
docker build -t titanic-app .

# Lancement du container (✅ réussie) 
docker run -p 8501:8501 titanic-app
