# app.py — Loan Default Scoring (seuils uniquement là où c'est indispensable)
# - PAS de seuil pour: total_debt_outstanding, income, years_employed, loan_amt_outstanding
# - Seuils "logiques" pour: fico_score (300..850), credit_lines_outstanding (>=0), customer_id (>=0)
# - Saisie texte (virgule/point), conversion/validation, integer si nécessaire
# - CSV optionnel pour typer les champs et fournir des valeurs par défaut

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st


# ============================ Paramètres ============================

MODEL_PKL  = os.getenv("MODEL_PKL", "model.pkl")
DATA_PATH  = os.getenv("DATA_PATH", "data/Loan_Data.csv")  # laisse tel quel si tu mets ton CSV dans data/
TARGET_COL = os.getenv("TARGET_COL", "default")
DEF_THRESH = float(os.getenv("THRESHOLD", "0.5"))

st.set_page_config(page_title="Loan Default — Scoring", layout="centered")

# Colonnes pour lesquelles on NE MET AUCUN SEUIL (pas de min/max)
NO_BOUNDS: set[str] = {
    "total_debt_outstanding",
    "income",
    "years_employed",
    "loan_amt_outstanding",
    "loan_amt_oustanding",  # tolérance à l'orthographe variante
}

# Contraintes "dures" UNIQUEMENT là où c'est indispensable (min, max) ; None = pas de borne
HARD_BOUNDS: Dict[str, Tuple[float | None, float | None]] = {
    "fico_score": (300, 850),             # domaine FICO standard
    "credit_lines_outstanding": (0, None),# pas de négatif
    "customer_id": (0, None),             # pas de négatif
    # Ajoute ici d'autres colonnes si nécessaire, ex: "age": (0, 120)
}

# Colonnes à traiter comme entières (si on n'a pas le CSV pour le déduire)
LIKELY_INT_COLS: set[str] = {
    "credit_lines_outstanding", "years_employed", "fico_score", "customer_id"
}


# ============================ Utilitaires ===========================

@st.cache_resource(show_spinner=False)
def load_model(pkl_path: str):
    """Charge le modèle pickle (chemins relatifs acceptés)."""
    p = Path(pkl_path)
    if not p.exists():
        base = Path(__file__).resolve().parent
        cands = [(base / pkl_path).resolve(), (base / ".." / pkl_path).resolve()]
        for c in cands:
            if c.exists():
                p = c
                break
    if not p.exists():
        raise FileNotFoundError(f"Modèle introuvable: {pkl_path}")
    model = joblib.load(p)
    return model, str(p)

def try_load_df(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Charge un CSV si possible ; sinon renvoie None (l'app reste fonctionnelle)."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        base = Path(__file__).resolve().parent
        cands = [(base / path).resolve(), (base / ".." / path).resolve()]
        for c in cands:
            if c.exists():
                p = c
                break
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    """Renvoie une proba de classe 1, quelle que soit l'API du modèle."""
    # 1) predict_proba
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        pass
    # 2) decision_function -> sigmoïde
    try:
        from scipy.special import expit
        s = model.decision_function(X)
        return expit(s)
    except Exception:
        pass
    # 3) predict (éventuellement 0/1)
    y = model.predict(X)
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] > 1:
        y = y[:, 1]
    return y

def numeric_text_input(label: str, default: float,
                       min_value: float | None = None,
                       max_value: float | None = None,
                       integer: bool = False) -> float:
    """
    Champ texte 'numérique' : accepte virgule et point ; valide min/max si fournis.
    Si integer=True, arrondit à l'entier (utile pour fico_score, credit_lines_outstanding, etc.).
    """
    raw = st.text_input(label, value=f"{default}")
    s = raw.strip().replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        st.warning(f"« {label} » doit être un nombre (ex : 0,25).")
        st.stop()
    if min_value is not None and val < min_value:
        st.warning(f"« {label} » doit être ≥ {min_value}.")
        st.stop()
    if max_value is not None and val > max_value:
        st.warning(f"« {label} » doit être ≤ {max_value}.")
        st.stop()
    if integer:
        val = float(int(round(val)))
    return val


# ============================ Chargements ===========================

try:
    model, resolved_model_path = load_model(MODEL_PKL)
except Exception as e:
    st.error(f"❌ Impossible de charger le modèle: {e}")
    st.stop()

df = try_load_df(DATA_PATH)

# Déterminer les features (priorité au modèle)
if hasattr(model, "feature_names_in_") and len(getattr(model, "feature_names_in_")) > 0:
    feature_cols: List[str] = list(getattr(model, "feature_names_in_"))
else:
    if df is None:
        st.error("❌ Ni dataset ni feature_names_in_ : impossible de déduire les variables d'entrée.")
        st.stop()
    if TARGET_COL not in df.columns:
        st.error(f"❌ Colonne cible « {TARGET_COL} » absente du CSV.")
        st.stop()
    feature_cols = [c for c in df.columns if c != TARGET_COL]


# ================================ UI =================================

st.title("🏦 Loan Default — Scoring")
st.caption("Saisis librement tes valeurs (virgule ou point). Seules quelques variables ont des bornes logiques (FICO, comptes ≥ 0).")

with st.sidebar:
    st.header("Variables du modèle")
    st.write(f"**Modèle** : `{Path(resolved_model_path).name}`")
    if df is not None:
        st.write(f"**Dataset détecté** : `{Path(DATA_PATH).name}`")

    inputs: Dict[str, Any] = {}
    for col in feature_cols:
        # Cas avec dataset : on déduit type et valeur par défaut depuis le CSV
        if df is not None and col in df.columns:
            s = df[col].dropna()

            # Numérique (float/int)
            if pd.api.types.is_numeric_dtype(s):
                default = float(np.nanmedian(s)) if len(s) else 0.0
                is_int = pd.api.types.is_integer_dtype(df[col]) or (col in LIKELY_INT_COLS)

                # Seuils UNIQUEMENT si indispensables
                if col in NO_BOUNDS:
                    minv, maxv = None, None
                else:
                    minv, maxv = HARD_BOUNDS.get(col, (None, None))

                inputs[col] = numeric_text_input(col, default, minv, maxv, integer=is_int)

            # Booléen (dtype bool ou valeurs {0,1,True,False})
            elif pd.api.types.is_bool_dtype(s) or set(map(str, s.unique())).issubset({"0", "1", "True", "False"}):
                mode_bool = False
                try:
                    mode_bool = bool(int(pd.Series(s).mode().iat[0]))
                except Exception:
                    pass
                inputs[col] = st.selectbox(col, options=[False, True], index=int(mode_bool))

            # Catégorielle (peu de modalités)
            elif s.nunique() <= 25:
                choices = sorted(map(str, s.unique().tolist()))
                default_choice = str(pd.Series(s).mode().iat[0]) if len(s) else choices[0]
                inputs[col] = st.selectbox(col, choices, index=choices.index(default_choice))

            # Texte libre sinon
            else:
                default_txt = str(pd.Series(s).mode().iat[0]) if len(s) else ""
                inputs[col] = st.text_input(col, value=default_txt)

        # Cas sans dataset : suppose numérique, avec contraintes si nécessaires
        else:
            is_int = col in LIKELY_INT_COLS
            if col in NO_BOUNDS:
                minv, maxv = None, None
            else:
                minv, maxv = HARD_BOUNDS.get(col, (None, None))
            inputs[col] = numeric_text_input(col, 0.0, minv, maxv, integer=is_int)

    threshold = st.slider("Seuil de décision", 0.0, 1.0, float(DEF_THRESH), 0.01)

with st.form("predict-form"):
    st.subheader("Paramètres saisis")
    st.write(pd.DataFrame([inputs], columns=feature_cols))
    submitted = st.form_submit_button("🔮 Prédire")

if submitted:
    X = pd.DataFrame([inputs], columns=feature_cols)

    # Conversion en numérique pour les colonnes numériques du CSV (si présent)
    if df is not None:
        for c in feature_cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                try:
                    X[c] = pd.to_numeric(pd.Series(X[c]).astype(str).str.replace(",", "."), errors="coerce")
                except Exception:
                    pass

    try:
        proba = float(predict_proba_safe(model, X)[0])
        pred = int(proba >= threshold)
        st.success("Prédiction effectuée ✅")
        c1, c2 = st.columns(2)
        c1.metric("Probabilité (classe 1)", f"{proba:.3f}")
        c2.metric("Décision", "1 (défaut)" if pred == 1 else "0 (non défaut)")
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")

with st.expander("ℹ️ Détails"):
    st.write(f"Chemin modèle : `{resolved_model_path}`")
    st.write(f"Variables : {len(feature_cols)}")
    st.write("Sans seuils sur : " + (", ".join(sorted(NO_BOUNDS)) or "—"))
    if HARD_BOUNDS:
        st.write("Contraintes indispensables :", HARD_BOUNDS)
