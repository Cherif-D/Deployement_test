# app.py — Loan Default Scoring (Design Amélioré)
# - Design professionnel centré sur l'utilisateur
# - Regroupement logique des champs dans un formulaire principal
# - Barre latérale pour les options et les méta-informations
# - Résultat visuel (couleurs, jauge) pour une interprétation rapide

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go


# ============================ Paramètres & Configuration ============================

# Configuration de la page (doit être la première commande st)
st.set_page_config(
    page_title="Évaluation de Risque de Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Variables d'environnement et constantes ---
MODEL_PKL  = os.getenv("MODEL_PKL", "model.pkl")
DATA_PATH  = os.getenv("DATA_PATH", "data/Loan_Data.csv")
TARGET_COL = os.getenv("TARGET_COL", "default")
DEF_THRESH = float(os.getenv("THRESHOLD", "0.5"))

# --- Règles de validation des champs ---
NO_BOUNDS: set[str] = {
    "total_debt_outstanding", "income", "years_employed",
    "loan_amt_outstanding", "loan_amt_oustanding",
}
HARD_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "fico_score": (300, 850),
    "credit_lines_outstanding": (0, None),
    "customer_id": (0, None),
}
LIKELY_INT_COLS: set[str] = {
    "credit_lines_outstanding", "years_employed", "fico_score", "customer_id"
}


# ============================ Style CSS Personnalisé ===========================

st.markdown("""
<style>
    /* Conteneur principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Style des cartes de résultat */
    .result-card {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .result-card h3 {
        color: white;
        font-size: 24px;
        margin-bottom: 5px;
    }
    .result-card p {
        font-size: 18px;
        margin-top: 5px;
    }
    .low-risk {
        background: linear-gradient(135deg, #28a745, #218838); /* Vert */
    }
    .high-risk {
        background: linear-gradient(135deg, #dc3545, #c82333); /* Rouge */
    }
    .medium-risk {
        background: linear-gradient(135deg, #ffc107, #e0a800); /* Orange */
    }
    /* Amélioration des formulaires */
    .stForm {
        border: 1px solid #262730; /* Bordure discrète pour le mode sombre */
        border-radius: 10px;
        padding: 20px;
    }
    [data-theme="light"] .stForm {
        border: 1px solid #e1e4e8; /* Bordure discrète pour le mode clair */
    }
</style>
""", unsafe_allow_html=True)


# ============================ Utilitaires ===========================

@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(pkl_path: str) -> Tuple[Any, str]:
    """Charge le modèle pickle (gère les chemins relatifs)."""
    p = Path(pkl_path)
    if not p.exists():
        base = Path(__file__).resolve().parent
        cands = [(base / pkl_path).resolve(), (base / ".." / pkl_path).resolve()]
        p = next((c for c in cands if c.exists()), p)
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable aux emplacements vérifiés : {pkl_path}")
    return joblib.load(p), str(p)

@st.cache_data
def try_load_df(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Charge un CSV si possible ; sinon renvoie None."""
    if not path: return None
    p = Path(path)
    if not p.is_absolute():
        base = Path(__file__).resolve().parent
        cands = [(base / path).resolve(), (base / ".." / path).resolve()]
        p = next((c for c in cands if c.exists()), p)
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    """Renvoie une proba de classe 1, quelle que soit l'API du modèle."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X))
    y = model.predict(X)
    return np.asarray(y, dtype=float).flatten()

def numeric_text_input(label: str, default: float, key: str, **kwargs) -> float:
    """Champ texte 'numérique' qui accepte virgule/point et valide les bornes."""
    min_val, max_val = kwargs.get("min_value"), kwargs.get("max_value")
    is_int = kwargs.get("integer", False)

    raw = st.text_input(label, value=f"{default}", key=key)
    s = raw.strip().replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        st.error(f"'{label}' doit être un nombre valide (ex: 1234.56).", icon="🚨")
        st.stop()

    if min_val is not None and val < min_val:
        st.error(f"'{label}' doit être supérieur ou égal à {min_val}.", icon="🔼")
        st.stop()
    if max_val is not None and val > max_val:
        st.error(f"'{label}' doit être inférieur ou égal à {max_val}.", icon="🔽")
        st.stop()

    return float(int(round(val))) if is_int else val

def create_score_gauge(score: float, threshold: float) -> go.Figure:
    """Crée une jauge de score avec Plotly."""
    if score >= threshold:
        color = "#dc3545" # Rouge
        text = "Risque Élevé"
    else:
        color = "#28a745" # Vert
        text = "Risque Faible"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number={'suffix': "%", 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': text, 'font': {'size': 24, 'color': color}},
        delta={'reference': threshold * 100, 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold * 100], 'color': 'rgba(40, 167, 69, 0.2)'},
                {'range': [threshold * 100, 100], 'color': 'rgba(220, 53, 69, 0.2)'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ============================ Chargements & Initialisation ===========================

try:
    model, resolved_model_path = load_model(MODEL_PKL)
except Exception as e:
    st.error(f"**Erreur Critique :** Impossible de charger le modèle. Vérifiez le chemin `{MODEL_PKL}`.\n\n*Détail : {e}*", icon="❌")
    st.stop()

df_data = try_load_df(DATA_PATH)

if hasattr(model, "feature_names_in_"):
    feature_cols: List[str] = list(model.feature_names_in_)
elif df_data is not None:
    feature_cols = [c for c in df_data.columns if c != TARGET_COL]
else:
    st.error("**Erreur Critique :** Impossible de déterminer les variables d'entrée.", icon="❌")
    st.stop()


# ================================ Interface Utilisateur =================================

st.title("🏦 Évaluateur de Risque de Crédit")
st.markdown("Entrez les informations du demandeur pour obtenir un score de probabilité de défaut de paiement.")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("⚙️ Options")
    threshold = st.slider("Seuil de décision", 0.0, 1.0, float(DEF_THRESH), 0.01,
                            help="Probabilité au-dessus de laquelle le statut est 'Défaut'.")

    st.markdown("---")
    st.header("Informations du Modèle")
    st.info(f"**Modèle chargé** :\n`{Path(resolved_model_path).name}`")
    if df_data is not None:
        st.success(f"**Données de référence** :\n`{Path(DATA_PATH).name}`")
    else:
        st.warning("**Aucun fichier de données** de référence n'a été trouvé. Les valeurs par défaut sont à 0.")

# --- FORMULAIRE DE SAISIE PRINCIPAL ---
inputs: Dict[str, Any] = {}
with st.form("loan_application_form"):
    st.header("📝 Formulaire de demande")

    # Organisation en colonnes pour une meilleure lisibilité
    col1, col2, col3 = st.columns(3)

    # Répartir les champs dans les colonnes
    # (Cette répartition manuelle est plus robuste qu'une boucle pour le design)
    fields_per_col = (len(feature_cols) + 2) // 3
    feature_chunks = [
        feature_cols[i:i + fields_per_col]
        for i in range(0, len(feature_cols), fields_per_col)
    ]

    cols_widgets = [col1, col2, col3]

    for i, chunk in enumerate(feature_chunks):
        with cols_widgets[i]:
            for col in chunk:
                # Logique pour déterminer le type de champ et les valeurs par défaut
                if df_data is not None and col in df_data.columns:
                    s = df_data[col].dropna()
                    if pd.api.types.is_numeric_dtype(s):
                        default = float(s.median()) if not s.empty else 0.0
                        is_int = pd.api.types.is_integer_dtype(df_data[col]) or (col in LIKELY_INT_COLS)
                        minv, maxv = (None, None) if col in NO_BOUNDS else HARD_BOUNDS.get(col, (None, None))
                        inputs[col] = numeric_text_input(col, default, key=col, integer=is_int, min_value=minv, max_value=maxv)
                    elif s.nunique() <= 2: # Booléen / Binaire
                        default_bool = bool(s.mode()[0]) if not s.empty else False
                        inputs[col] = st.radio(col, [False, True], index=int(default_bool), horizontal=True, key=col)
                    elif s.nunique() <= 25: # Catégoriel
                        choices = sorted(map(str, s.unique().tolist()))
                        default_choice = str(s.mode()[0]) if not s.empty else choices[0]
                        inputs[col] = st.selectbox(col, choices, index=choices.index(default_choice), key=col)
                    else: # Texte
                        default_txt = str(s.mode()[0]) if not s.empty else ""
                        inputs[col] = st.text_input(col, value=default_txt, key=col)
                else: # Fallback si pas de CSV
                    is_int = col in LIKELY_INT_COLS
                    minv, maxv = (None, None) if col in NO_BOUNDS else HARD_BOUNDS.get(col, (None, None))
                    inputs[col] = numeric_text_input(col, 0.0, key=col, integer=is_int, min_value=minv, max_value=maxv)

    st.markdown("---")
    submitted = st.form_submit_button("🔮 Analyser le Dossier", use_container_width=True)


# --- SECTION DES RÉSULTATS ---
if submitted:
    st.header("✅ Résultats de l'Analyse")
    X = pd.DataFrame([inputs], columns=feature_cols)

    # Conversion de type finale
    if df_data is not None:
        for c in feature_cols:
            if c in df_data.columns and pd.api.types.is_numeric_dtype(df_data[c]):
                X[c] = pd.to_numeric(X[c], errors='coerce')

    try:
        proba = float(predict_proba_safe(model, X)[0])
        pred = int(proba >= threshold)

        res_col1, res_col2 = st.columns([1, 1.5])

        with res_col1:
            if pred == 1:
                verdict_text = "Risque Élevé"
                reco_text = "Défaut de paiement probable"
                css_class = "high-risk"
            else:
                verdict_text = "Risque Faible"
                reco_text = "Dossier favorable"
                css_class = "low-risk"

            st.markdown(f"""
            <div class="result-card {css_class}">
                <h3>{verdict_text}</h3>
                <p>{reco_text}</p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Score de Probabilité (Défaut)", f"{proba:.2%}")
            st.caption(f"Calculé avec un seuil de décision à {threshold:.0%}")


        with res_col2:
            st.plotly_chart(create_score_gauge(proba, threshold), use_container_width=True)

        with st.expander("📂 Voir les données saisies"):
            st.dataframe(X)

    except Exception as e:
        st.error(f"**Erreur lors de la prédiction :**\n\n`{e}`", icon="🔥")
