#!/usr/bin/env python3
"""
Application Streamlit - Détection d'Intrusion Réseau
=====================================================
Interface de prédiction en temps réel pour classifier
le trafic réseau (normal vs attaque) avec le dataset NSL-KDD.

Auteur: [Votre Nom]
Date: Mai 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="IDS - Détection d'Intrusion Réseau",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a237e, #283593);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-normal {
        background-color: #c8e6c9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        text-align: center;
    }
    .prediction-attack {
        background-color: #ffcdd2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #c62828;
        text-align: center;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .attack-detail {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #e65100;
    }
    .stApp {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTES ET CONFIGURATION
# ==========================================
MODEL_DIR = "models"
TRAIN_PATH = "KDDTrain+.txt"
TEST_PATH = "KDDTest+.txt"

COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

CAT_FEATURES = ['protocol_type', 'service', 'flag']

ATTACK_CATEGORIES = {
    'DoS': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
            'apache2', 'udpstorm', 'processtable', 'worm'],
    'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
    'R2L': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop',
            'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop',
            'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'],
    'U2R': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl',
            'sqlattack', 'xterm', 'ps']
}

def get_attack_category(attack):
    """Retourne la catégorie d'attaque."""
    for cat, attacks in ATTACK_CATEGORIES.items():
        if attack in attacks:
            return cat
    return 'normal'

# ==========================================
# CHARGEMENT / ENTRAÎNEMENT DES MODÈLES
# ==========================================
@st.cache_resource
def load_models():
    """Charge ou entraîne les modèles nécessaire."""
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    target_enc_path = os.path.join(MODEL_DIR, 'target_encoder.pkl')
    label_enc_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')

    if all(os.path.exists(p) for p in [model_path, scaler_path, target_enc_path, label_enc_path]):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        target_encoder = joblib.load(target_enc_path)
        label_encoders = joblib.load(label_enc_path)
        return model, scaler, target_encoder, label_encoders
    else:
        st.warning("Modèles pré-entraînés non trouvés. Entraînement en cours...")
        return train_models()

def train_models():
    """Entraîne les modèles à partir des fichiers CSV."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE

    # Chargement
    df_train = pd.read_csv(TRAIN_PATH, names=COL_NAMES)
    df_test = pd.read_csv(TEST_PATH, names=COL_NAMES)

    # Cible binaire
    df_train['binary_class'] = df_train['attack_type'].apply(
        lambda x: 'normal' if x == 'normal' else 'attack'
    )
    df_test['binary_class'] = df_test['attack_type'].apply(
        lambda x: 'normal' if x == 'normal' else 'attack'
    )

    X_train_raw = df_train[COL_NAMES[:41]].copy()
    y_train_raw = df_train['binary_class']
    X_test_raw = df_test[COL_NAMES[:41]].copy()
    y_test_raw = df_test['binary_class']

    # Encodage
    label_encoders = {}
    for col in CAT_FEATURES:
        le = LabelEncoder()
        combined = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).unique()
        le.fit(combined)
        X_train_raw[col] = le.transform(X_train_raw[col])
        X_test_raw[col] = le.transform(X_test_raw[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train_raw)
    y_test = target_encoder.transform(y_test_raw)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Entraînement
    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=42
    )
    model.fit(X_train_res, y_train_res)

    # Sauvegarde
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, 'target_encoder.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))

    st.success("Modèle entraîné et sauvegardé avec succès !")
    return model, scaler, target_encoder, label_encoders

# ==========================================
# FONCTIONS DE PRÉDICTION
# ==========================================
def preprocess_input(df, scaler, label_encoders):
    """Prétraite les données d'entrée pour la prédiction."""
    df = df.copy()

    # Vérifier les colonnes
    missing_cols = [c for c in COL_NAMES[:41] if c not in df.columns]
    if missing_cols:
        st.error(f"Colonnes manquantes : {missing_cols}")
        return None

    df = df[COL_NAMES[:41]]

    # Encodage
    for col in CAT_FEATURES:
        le = label_encoders[col]
        # Gérer les valeurs inconnues
        known_classes = set(le.classes_)
        df[col] = df[col].apply(
            lambda x: x if x in known_classes else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    # Normalisation
    df_scaled = scaler.transform(df)
    return df_scaled

def predict_with_details(model, scaler, target_encoder, label_encoders, df_input):
    """Effectue la prédiction avec détails."""
    processed = preprocess_input(df_input, scaler, label_encoders)
    if processed is None:
        return None, None, None

    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)

    pred_labels = target_encoder.inverse_transform(predictions)
    confidences = np.max(probabilities, axis=1)

    return pred_labels, confidences, predictions

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Système de Détection d'Intrusion Réseau</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            Classification automatique du trafic réseau (Normal vs Attaque)
            basée sur le dataset NSL-KDD
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chargement des modèles
    with st.spinner("Chargement des modèles..."):
        model, scaler, target_encoder, label_encoders = load_models()

    # Barre latérale
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=80)
        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["📤 Prédiction CSV", "📝 Prédiction Manuelle", "📊 Visualisations", "ℹ️ À propos"]
        )

        st.markdown("---")
        st.markdown("### Modèle actif")
        st.info("Random Forest\n(n_estimators=100, max_depth=20)")
        st.markdown("### Dataset")
        st.caption("NSL-KDD")
        st.caption(f"Entraîné sur KDDTrain+.txt")

    # ==========================================
    # PAGE 1: Prédiction par upload CSV
    # ==========================================
    if page == "📤 Prédiction CSV":
        st.markdown("## 📤 Prédiction par fichier CSV")
        st.markdown("""
        Téléchargez un fichier CSV contenant les 41 caractéristiques réseau.
        Le modèle analysera chaque ligne et prédira s'il s'agit de trafic normal ou d'une attaque.
        """)

        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV",
            type=['csv', 'txt'],
            help="Le fichier doit contenir les 41 colonnes du dataset NSL-KDD"
        )

        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file, header=None, names=COL_NAMES[:41])
            except:
                df_input = pd.read_csv(uploaded_file)

            st.markdown(f"**Fichier chargé :** {uploaded_file.name}")
            st.markdown(f"**Lignes :** {len(df_input)}")
            st.markdown(f"**Colonnes :** {len(df_input.columns)}")

            with st.expander("Aperçu des données", expanded=False):
                st.dataframe(df_input.head(10), use_container_width=True)

            if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    pred_labels, confidences, _ = predict_with_details(
                        model, scaler, target_encoder, label_encoders, df_input
                    )

                if pred_labels is not None:
                    # Résumé
                    n_normal = sum(pred_labels == 'normal')
                    n_attack = sum(pred_labels == 'attack')

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Total</h3>
                            <p style="font-size: 2rem; font-weight: bold;">{len(pred_labels)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #2e7d32;">✅ Normal</h3>
                            <p style="font-size: 2rem; font-weight: bold; color: #2e7d32;">{n_normal}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="color: #c62828;">⚠️ Attaque</h3>
                            <p style="font-size: 2rem; font-weight: bold; color: #c62828;">{n_attack}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Résultats détaillés
                    st.markdown("---")
                    st.markdown("### Résultats détaillés")

                    results_df = df_input.copy()
                    results_df['Prédiction'] = pred_labels
                    results_df['Confiance'] = [f"{c*100:.1f}%" for c in confidences]

                    # Colorer les lignes
                    def color_row(row):
                        if row['Prédiction'] == 'attack':
                            return ['background-color: #ffebee'] * len(row)
                        return ['background-color: #e8f5e9'] * len(row)

                    st.dataframe(
                        results_df.style.apply(color_row, axis=1),
                        use_container_width=True,
                        height=400
                    )

                    # Graphique
                    fig, ax = plt.subplots(figsize=(8, 5))
                    counts = pd.Series(pred_labels).value_counts()
                    colors = ['#4CAF50' if x == 'normal' else '#F44336' for x in counts.index]
                    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.5)
                    for bar, val in zip(bars, counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                str(val), ha='center', fontweight='bold', fontsize=14)
                    ax.set_title('Résultats de la Prédiction', fontweight='bold', fontsize=14)
                    ax.set_ylabel('Nombre d\'occurrences')
                    ax.set_xlabel('Classe prédite')
                    st.pyplot(fig)

    # ==========================================
    # PAGE 2: Prédiction manuelle
    # ==========================================
    elif page == "📝 Prédiction Manuelle":
        st.markdown("## 📝 Prédiction Manuelle")
        st.markdown("""
        Saisissez les caractéristiques d'une connexion réseau pour
        déterminer si elle est normale ou malveillante.
        """)

        # Initialiser les valeurs par défaut dans session_state
        if 'example_mode' not in st.session_state:
            st.session_state.example_mode = None

        with st.expander("🔄 Remplir avec des valeurs par défaut", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Exemple Normal"):
                    st.session_state.example_mode = 'normal'
                    st.rerun()
            with col2:
                if st.button("Exemple Attaque (DoS)"):
                    st.session_state.example_mode = 'attack'
                    st.rerun()

        # Définir les valeurs selon le mode sélectionné
        if st.session_state.example_mode == 'normal':
            dv = {
                'duration': 0, 'protocol_type': 'tcp', 'service': 'http',
                'flag': 'SF', 'src_bytes': 181, 'dst_bytes': 5450,
                'count': 1, 'srv_count': 1, 'same_srv_rate': 1.0,
                'diff_srv_rate': 0.0, 'dst_host_count': 5, 'dst_host_srv_count': 5
            }
        elif st.session_state.example_mode == 'attack':
            dv = {
                'duration': 0, 'protocol_type': 'tcp', 'service': 'private',
                'flag': 'S0', 'src_bytes': 0, 'dst_bytes': 0,
                'count': 123, 'srv_count': 6, 'same_srv_rate': 0.0,
                'diff_srv_rate': 1.0, 'dst_host_count': 255, 'dst_host_srv_count': 26
            }
        else:
            dv = {}

        # Listes de choix pour les selectbox
        protocols = ['tcp', 'udp', 'icmp']
        services = ['http', 'ftp_data', 'private', 'smtp', 'telnet', 'ssh', 'other']
        flags = ['SF', 'S0', 'S1', 'S2', 'S3', 'REJ', 'RSTO', 'RSTR', 'OTH', 'SH']

        # Interface de saisie en 3 colonnes
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🖥️ Informations de base**")
            duration = st.number_input('Duration', min_value=0, value=dv.get('duration', 0), help="Durée de la connexion")
            protocol = st.selectbox('Protocol Type', protocols,
                                    index=protocols.index(dv['protocol_type']) if 'protocol_type' in dv else 0,
                                    help="Type de protocole")
            service = st.selectbox('Service', services,
                                   index=services.index(dv['service']) if 'service' in dv else 0,
                                   help="Service réseau")
            flag = st.selectbox('Flag', flags,
                                index=flags.index(dv['flag']) if 'flag' in dv else 0,
                                help="État de la connexion")
            src_bytes = st.number_input('Src Bytes', min_value=0, value=dv.get('src_bytes', 0), help="Octets source")
            dst_bytes = st.number_input('Dst Bytes', min_value=0, value=dv.get('dst_bytes', 0), help="Octets destination")
            land = st.selectbox('Land', [0, 1], index=0, help="Même hôte source/dest")
            wrong_fragment = st.number_input('Wrong Fragment', min_value=0, value=0)
            urgent = st.number_input('Urgent', min_value=0, value=0)

        with col2:
            st.markdown("**🔐 Informations de connexion**")
            hot = st.number_input('Hot', min_value=0, value=0)
            num_failed_logins = st.number_input('Num Failed Logins', min_value=0, value=0)
            logged_in = st.selectbox('Logged In', [0, 1], index=1)
            num_compromised = st.number_input('Num Compromised', min_value=0, value=0)
            root_shell = st.selectbox('Root Shell', [0, 1], index=0)
            su_attempted = st.selectbox('Su Attempted', [0, 1], index=0)
            num_root = st.number_input('Num Root', min_value=0, value=0)
            num_file_creations = st.number_input('Num File Creations', min_value=0, value=0)
            num_shells = st.number_input('Num Shells', min_value=0, value=0)
            num_access_files = st.number_input('Num Access Files', min_value=0, value=0)
            num_outbound_cmds = st.number_input('Num Outbound Cmds', min_value=0, value=0)
            is_host_login = st.selectbox('Is Host Login', [0, 1], index=0)
            is_guest_login = st.selectbox('Is Guest Login', [0, 1], index=0)

        with col3:
            st.markdown("**🌐 Trafic réseau**")
            count = st.number_input('Count', min_value=0, value=dv.get('count', 1), help="Connexions vers même hôte")
            srv_count = st.number_input('Srv Count', min_value=0, value=dv.get('srv_count', 1))
            serror_rate = st.slider('Serror Rate', 0.0, 1.0, 0.0, help="Taux d'erreurs SYN")
            srv_serror_rate = st.slider('Srv Serror Rate', 0.0, 1.0, 0.0)
            rerror_rate = st.slider('Rerror Rate', 0.0, 1.0, 0.0)
            srv_rerror_rate = st.slider('Srv Rerror Rate', 0.0, 1.0, 0.0)
            same_srv_rate = st.slider('Same Srv Rate', 0.0, 1.0, dv.get('same_srv_rate', 1.0))
            diff_srv_rate = st.slider('Diff Srv Rate', 0.0, 1.0, dv.get('diff_srv_rate', 0.0))
            srv_diff_host_rate = st.slider('Srv Diff Host Rate', 0.0, 1.0, 0.0)

            st.markdown("**🏠 Hôte destinataire**")
            dst_host_count = st.number_input('Dst Host Count', min_value=0, value=dv.get('dst_host_count', 5))
            dst_host_srv_count = st.number_input('Dst Host Srv Count', min_value=0, value=dv.get('dst_host_srv_count', 5))
            dst_host_same_srv_rate = st.slider('Dst Host Same Srv Rate', 0.0, 1.0, 1.0)
            dst_host_diff_srv_rate = st.slider('Dst Host Diff Srv Rate', 0.0, 1.0, 0.0)
            dst_host_same_src_port_rate = st.slider('Dst Host Same Src Port Rate', 0.0, 1.0, 0.0)
            dst_host_srv_diff_host_rate = st.slider('Dst Host Srv Diff Host Rate', 0.0, 1.0, 0.0)
            dst_host_serror_rate = st.slider('Dst Host Serror Rate', 0.0, 1.0, 0.0)
            dst_host_srv_serror_rate = st.slider('Dst Host Srv Serror Rate', 0.0, 1.0, 0.0)
            dst_host_rerror_rate = st.slider('Dst Host Rerror Rate', 0.0, 1.0, 0.0)
            dst_host_srv_rerror_rate = st.slider('Dst Host Srv Rerror Rate', 0.0, 1.0, 0.0)

        # Bouton de prédiction
        st.markdown("---")
        if st.button("🔍 Analyser la connexion", type="primary", use_container_width=True):
            # Construire le DataFrame
            input_dict = {
                'duration': [duration], 'protocol_type': [protocol], 'service': [service],
                'flag': [flag], 'src_bytes': [src_bytes], 'dst_bytes': [dst_bytes],
                'land': [land], 'wrong_fragment': [wrong_fragment], 'urgent': [urgent],
                'hot': [hot], 'num_failed_logins': [num_failed_logins], 'logged_in': [logged_in],
                'num_compromised': [num_compromised], 'root_shell': [root_shell],
                'su_attempted': [su_attempted], 'num_root': [num_root],
                'num_file_creations': [num_file_creations], 'num_shells': [num_shells],
                'num_access_files': [num_access_files], 'num_outbound_cmds': [num_outbound_cmds],
                'is_host_login': [is_host_login], 'is_guest_login': [is_guest_login],
                'count': [count], 'srv_count': [srv_count],
                'serror_rate': [serror_rate], 'srv_serror_rate': [srv_serror_rate],
                'rerror_rate': [rerror_rate], 'srv_rerror_rate': [srv_rerror_rate],
                'same_srv_rate': [same_srv_rate], 'diff_srv_rate': [diff_srv_rate],
                'srv_diff_host_rate': [srv_diff_host_rate],
                'dst_host_count': [dst_host_count], 'dst_host_srv_count': [dst_host_srv_count],
                'dst_host_same_srv_rate': [dst_host_same_srv_rate],
                'dst_host_diff_srv_rate': [dst_host_diff_srv_rate],
                'dst_host_same_src_port_rate': [dst_host_same_src_port_rate],
                'dst_host_srv_diff_host_rate': [dst_host_srv_diff_host_rate],
                'dst_host_serror_rate': [dst_host_serror_rate],
                'dst_host_srv_serror_rate': [dst_host_srv_serror_rate],
                'dst_host_rerror_rate': [dst_host_rerror_rate],
                'dst_host_srv_rerror_rate': [dst_host_srv_rerror_rate]
            }

            df_input = pd.DataFrame(input_dict)

            with st.spinner("Analyse en cours..."):
                pred_label, confidence, pred_raw = predict_with_details(
                    model, scaler, target_encoder, label_encoders, df_input
                )

            if pred_label is not None:
                st.markdown("---")
                st.markdown("### Résultat de l'analyse")

                if pred_label[0] == 'normal':
                    st.markdown(f"""
                    <div class="prediction-normal">
                        <h2 style="margin: 0;">✅ Trafic NORMAL</h2>
                        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                            Confiance : {confidence[0]*100:.2f}%
                        </p>
                        <p style="margin-top: 0.5rem; color: #555;">
                            Aucune menace détectée. La connexion semble légitime.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-attack">
                        <h2 style="margin: 0;">🚨 ATTAQUE DÉTECTÉE</h2>
                        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                            Confiance : {confidence[0]*100:.2f}%
                        </p>
                        <p style="margin-top: 0.5rem; color: #555;">
                            Trafic malveillant identifié. Des mesures de sécurité sont requises.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Analyse de la menace
                    st.markdown("### Analyse de la menace")
                    threat_score = confidence[0] * 100
                    if threat_score > 95:
                        st.error("🔴 Menace CRITIQUE - Probabilité d'attaque très élevée")
                    elif threat_score > 80:
                        st.warning("🟠 Menace ÉLEVÉE - Probabilité d'attaque significative")
                    else:
                        st.info("🟡 Menace MODÉRÉE - Surveiller la connexion")

    # ==========================================
    # PAGE 3: Visualisations
    # ==========================================
    elif page == "📊 Visualisations":
        st.markdown("## 📊 Visualisations des Données")
        st.markdown("Analyse exploratoire du dataset NSL-KDD.")

        if not os.path.exists("figures"):
            os.makedirs("figures", exist_ok=True)
            st.info("Générez les figures en exécutant d'abord le notebook Jupyter.")

        # Vérifier si les datasets existent
        if os.path.exists(TRAIN_PATH):
            df_train = pd.read_csv(TRAIN_PATH, names=COL_NAMES)
            df_train['binary_class'] = df_train['attack_type'].apply(
                lambda x: 'normal' if x == 'normal' else 'attack'
            )

            viz_option = st.selectbox(
                "Choisir une visualisation",
                [
                    "Distribution des classes",
                    "Types d'attaques",
                    "Matrice de corrélation",
                    "Top features importantes"
                ]
            )

            if viz_option == "Distribution des classes":
                fig, ax = plt.subplots(figsize=(8, 5))
                counts = df_train['binary_class'].value_counts()
                colors = ['#4CAF50', '#F44336']
                wedges, texts, autotexts = ax.pie(
                    counts.values, labels=counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90, explode=[0.03, 0.03]
                )
                ax.set_title('Distribution Normal vs Attaque', fontweight='bold', fontsize=14)
                st.pyplot(fig)

            elif viz_option == "Types d'attaques":
                df_train['attack_category'] = df_train['attack_type'].apply(get_attack_category)
                fig, ax = plt.subplots(figsize=(10, 6))
                cat_counts = df_train['attack_category'].value_counts()
                cat_colors = {'normal': '#4CAF50', 'DoS': '#F44336', 'Probe': '#FF9800',
                              'R2L': '#2196F3', 'U2R': '#9C27B0'}
                bars = ax.bar(cat_counts.index, cat_counts.values,
                              color=[cat_colors.get(x, '#888') for x in cat_counts.index],
                              edgecolor='black', width=0.5)
                for bar, val in zip(bars, cat_counts.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                            str(val), ha='center', fontweight='bold')
                ax.set_title('Distribution des Catégories d\'Attaques', fontweight='bold')
                ax.set_ylabel('Nombre d\'occurrences')
                st.pyplot(fig)

            elif viz_option == "Matrice de corrélation":
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns.drop(
                    ['difficulty'], errors='ignore'
                )
                fig, ax = plt.subplots(figsize=(18, 14))
                corr = df_train[numeric_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r',
                            center=0, square=True, linewidths=0.3, ax=ax)
                ax.set_title('Matrice de Corrélation', fontweight='bold', fontsize=14)
                st.pyplot(fig)

            elif viz_option == "Top features importantes":
                if os.path.exists(os.path.join(MODEL_DIR, 'random_forest_model.pkl')):
                    rf_model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
                    importances = rf_model.feature_importances_
                    indices = np.argsort(importances)[::-1][:15]

                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.barh(range(15), importances[indices][::-1],
                            color=sns.color_palette("Blues_r", 15))
                    ax.set_yticks(range(15))
                    ax.set_yticklabels([COL_NAMES[i] for i in indices][::-1])
                    ax.set_title('Top 15 Features Importantes', fontweight='bold')
                    ax.set_xlabel('Importance relative')
                    st.pyplot(fig)
                else:
                    st.warning("Entraînez d'abord le modèle via l'interface de prédiction.")

    # ==========================================
    # PAGE 4: À propos
    # ==========================================
    elif page == "ℹ️ À propos":
        st.markdown("## ℹ️ À propos du projet")
        st.markdown("""
        ### Projet de Détection d'Intrusion Réseau
        **Contexte :** Examen universitaire - Machine Learning & Cybersécurité

        **Objectif :** Développer un système de classification automatique
        capable de distinguer le trafic réseau normal des attaques.

        **Dataset :** NSL-KDD (version améliorée du KDD Cup 1999)
        - 41 caractéristiques réseau
        - 4 catégories d'attaques : DoS, Probe, R2L, U2R
        - 125 973 enregistrements d'entraînement
        - 22 544 enregistrements de test

        **Modèle utilisé :** Random Forest Classifier
        - n_estimators: 100
        - max_depth: 20
        - min_samples_split: 5
        - min_samples_leaf: 2

        **Performance attendue :**
        - Accuracy: > 99%
        - Précision: > 99%
        - Rappel: > 99%
        - F1-Score: > 99%

        **Améliorations possibles :**
        - GridSearchCV pour l'optimisation des hyperparamètres
        - XGBoost pour de meilleures performances
        - LSTM pour la détection de patterns temporels
        - Test sur des datasets modernes (CIC-IDS2017)
        """)

if __name__ == "__main__":
    main()
