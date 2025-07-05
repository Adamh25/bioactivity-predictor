import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Function to compute descriptors
# -------------------------------
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        return [mw, logp, h_donors, h_acceptors]
    else:
        return [np.nan, np.nan, np.nan, np.nan]

# -------------------------------
# Load and Clean Bioactivity Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bioactivity_data_Acetylcholinesterase.csv")  # Pre-cleaned ChEMBL dataset
    df['pIC50'] = -np.log10(df['IC50'] * 1e-9)  # Convert IC50 to pIC50
    descriptors = df['SMILES'].apply(compute_descriptors)
    descriptors_df = pd.DataFrame(descriptors.tolist(), columns=["MW", "LogP", "H_Donors", "H_Acceptors"])
    df = pd.concat([df, descriptors_df], axis=1)
    df.dropna(inplace=True)
    return df

# -------------------------------
# Streamlit Web Interface
# -------------------------------
st.title("Acetylcholinesterase Inhibitor Potency Prediction")
st.write("Upload chemical information (SMILES) to predict pIC50 values using machine learning.")

# Load and display dataset
df = load_data()
st.subheader("Bioactivity Dataset (Preview)")
st.dataframe(df.head())

# -------------------------------
# Data Visualization
# -------------------------------
st.subheader("Data Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="MW", y="pIC50", hue="LogP", ax=ax)
st.pyplot(fig)

# -------------------------------
# Model Training
# -------------------------------
st.subheader("Model Performance (Lazypredict)")
X = df[["MW", "LogP", "H_Donors", "H_Acceptors"]]
y = df["pIC50"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
st.dataframe(models)

# -------------------------------
# User Prediction Interface
# -------------------------------
st.subheader("Predict pIC50 for Your Molecule")

smiles_input = st.text_input("Enter SMILES notation:")
if st.button("Predict"):
    descriptors = compute_descriptors(smiles_input)
    if np.isnan(descriptors).any():
        st.error("Invalid SMILES string. Please check your input.")
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X, y)
        predicted_pIC50 = model.predict([descriptors])[0]
        st.success(f"Predicted pIC50: {predicted_pIC50:.2f}")


