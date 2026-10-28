import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NEXORA SCIENCE",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #164e63);
        border: 1px solid #334155;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 34px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
    }

    .footer {
        color: #94a3b8;
        font-size: 12px;
        text-align: center;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. FONCTIONS D'ANALYSE
# =========================================================

def lire_csv(contenu):
    """Lit un fichier CSV avec plusieurs encodages et séparateurs."""

    erreurs = []

    for encodage in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(contenu),
                sep=None,
                engine="python",
                encoding=encodage
            )

            if len(df.columns) == 1:
                for separateur in (";", ",", "\t", "|"):
                    essai = pd.read_csv(
                        io.BytesIO(contenu),
                        sep=separateur,
                        encoding=encodage
                    )

                    if len(essai.columns) > 1:
                        df = essai
                        break

            return df

        except Exception as erreur:
            erreurs.append(str(erreur))

    raise ValueError(
        erreurs[-1] if erreurs else "Format CSV non reconnu."
    )


def obtenir_numeriques(df):
    """Retourne les colonnes numériques utilisables."""

    numeriques = df.select_dtypes(include="number").copy()

    numeriques = numeriques.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeriques = numeriques.dropna(
        axis=1,
        how="all"
    )

    return numeriques


def detecter_anomalies(df):
    """Repère les valeurs inhabituelles selon la méthode IQR."""

    numeriques = obtenir_numeriques(df)
    anomalies = []

    for colonne in numeriques.columns:
        serie = numeriques[colonne].dropna()

        if len(serie) < 4:
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        limite_basse = q1 - 1.5 * iqr
        limite_haute = q3 + 1.5 * iqr

        masque = (
            (numeriques[colonne] < limite_basse)
            | (numeriques[colonne] > limite_haute)
        )

        positions = np.flatnonzero(
            masque.fillna(False).to_numpy()
        )

        for position in positions:
            anomalies.append({
                "Ligne du fichier": int(position + 2),
                "Variable": colonne,
                "Valeur observée": numeriques[colonne].iloc[position],
                "Limite basse": round(limite_basse, 3),
                "Limite haute": round(limite_haute, 3)
            })

    return pd.DataFrame(anomalies)


def generer_rapport(df, signature):
    """Prépare un rapport texte à télécharger."""

    numeriques = obtenir_numeriques(df)

    lignes = [
        "NEXORA SCIENCE — RAPPORT SCIENTIFIQUE",
        "=" * 45,
        "",
        f"Nombre de lignes : {len(df)}",
        f"Nombre de colonnes : {len(df.columns)}",
        f"Valeurs manquantes : {int(df.isna().sum().sum())}",
        ""
    ]

    if not numeriques.empty:
        lignes.extend([
            "STATISTIQUES DESCRIPTIVES",
            numeriques.describe().round(3).to_string(),
            ""
        ])
    else:
        lignes.extend([
            "Aucune colonne numérique détectée.",
            ""
        ])

    if numeriques.shape[1] >= 2:
        correlations = numeriques.corr()

        lignes.extend([
            "MATRICE DE CORRÉLATION",
            correlations.round(3).to_string(),
            ""
        ])

    anomalies = detecter_anomalies(df)

    lignes.append("VALEURS INHABITUELLES")

    if anomalies.empty:
        lignes.append(
            "Aucune valeur inhabituelle détectée par la méthode IQR."
        )
    else:
        lignes.append(anomalies.to_string(index=False))

    lignes.append("")

    resultat_ia = st.session_state.get("resultat_ia")

    if (
        resultat_ia is not None
        and resultat_ia.get("signature") == signature
    ):
        lignes.extend([
            "COMPARAISON DES MODÈLES D'IA",
            f"Variable prédite : {resultat_ia['cible']}",
            pd.DataFrame(
                resultat_ia["metriques"]
            ).to_string(index=False),
            ""
        ])

    lignes.extend([
        "LIMITES SCIENTIFIQUES",
        "Les corrélations ne démontrent pas une causalité.",
        "Les anomalies statistiques ne sont pas nécessairement des erreurs.",
        "La performance des modèles dépend de la qualité et de la quantité "
        "des données disponibles.",
        "Les résultats doivent être vérifiés avant toute conclusion."
    ])

    return "\n".join(lignes)


# =========================================================
# 3. NAVIGATION ET IMPORTATION
# =========================================================

with st.sidebar:
    st.title("🔬 NEXORA SCIENCE")
    st.caption("SCIENTIFIC DATA & AI")

    page = st.radio(
        "NAVIGATION",
        [
            "Tableau de bord",
            "Analyse des données",
            "Analyse scientifique",
            "Intelligence artificielle",
            "Rapports scientifiques"
        ]
    )

    st.divider()

    fichier = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    st.caption(
        "Importe des données que tu es autorisé à utiliser."
    )


df = None
signature = None

if fichier is not None:
    contenu = fichier.getvalue()
    signature = hashlib.sha256(contenu).hexdigest()

    if st.session_state.get("signature_donnees") != signature:
        st.session_state["signature_donnees"] = signature
        st.session_state.pop("resultat_ia", None)

    try:
        df = lire_csv(contenu)

        if df.empty:
            st.warning("Le fichier ne contient aucune observation.")
            df = None

    except Exception as erreur:
        st.error(f"Impossible de lire le fichier CSV : {erreur}")


# =========================================================
# 4. EN-TÊTE PROFESSIONNEL
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🔬 NEXORA SCIENCE</h1>
    <p>
        Analyse scientifique, visualisation des données
        et modélisation prédictive.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 5. TABLEAU DE BORD
# =========================================================

if page == "Tableau de bord":

    st.header("Tableau de bord")
    st.write(
        "Bienvenue dans NEXORA SCIENCE. "
        "Importe un fichier CSV dans le panneau latéral "
        "pour commencer."
    )

    if df is None:
        st.info("En attente d'un fichier CSV.")
    else:
        numeriques = obtenir_numeriques(df)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Observations", len(df))
        col2.metric("Variables", len(df.columns))
        col3.metric("Variables numériques", len(numeriques.columns))
        col4.metric(
            "Valeurs manquantes",
            int(df.isna().sum().sum())
        )

        st.subheader("Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Modules disponibles")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**📊 Analyse des données**")
            st.write("Statistiques descriptives et graphiques.")

            st.markdown("**🧪 Analyse scientifique**")
            st.write("Corrélations et détection des anomalies.")

        with c2:
            st.markdown("**🧠 Intelligence artificielle**")
            st.write("Entraînement et comparaison des modèles.")

            st.markdown("**📄 Rapports scientifiques**")
            st.write("Création et téléchargement des résultats.")


# =========================================================
# 6. ANALYSE DES DONNÉES
# =========================================================

elif page == "Analyse des données":

    st.header("📊 Analyse des données")

    if df is None:
        st.info("Importe un fichier CSV dans le panneau latéral.")

    else:
        numeriques = obtenir_numeriques(df)

        st.subheader("Aperçu du jeu de données")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Statistiques descriptives")

        if numeriques.empty:
            st.info("Aucune colonne numérique n'a été détectée.")
        else:
            st.dataframe(
                numeriques.describe().round(3),
                use_container_width=True
            )

            st.subheader("Visualisation")

            colonne = st.selectbox(
                "Choisis une variable numérique",
                list(numeriques.columns)
            )

            type_graphique = st.radio(
                "Type de graphique",
                ["Courbe", "Histogramme"],
                horizontal=True
            )

            if type_graphique == "Courbe":
                st.line_chart(
                    numeriques[colonne].reset_index(drop=True)
                )

            else:
                fig, ax = plt.subplots(figsize=(9, 4))

                ax.hist(
                    numeriques[colonne].dropna(),
                    bins="auto",
                    edgecolor="black"
                )

                ax.set_title(f"Distribution de {colonne}")
                ax.set_xlabel(colonne)
                ax.set_ylabel("Fréquence")

                st.pyplot(fig)
                plt.close(fig)


# =========================================================
# 7. ANALYSE SCIENTIFIQUE
# =========================================================

elif page == "Analyse scientifique":

    st.header("🧪 Analyse scientifique")

    if df is None:
        st.info("Importe un fichier CSV dans le panneau latéral.")

    else:
        numeriques = obtenir_numeriques(df)

        st.subheader("Corrélations")

        if numeriques.shape[1] < 2:
            st.info(
                "Il faut au moins deux colonnes numériques "
                "pour calculer les corrélations."
            )
        else:
            correlations = numeriques.corr()

            st.dataframe(
                correlations.round(3),
                use_container_width=True
            )

            fig, ax = plt.subplots(
                figsize=(
                    max(6, len(correlations.columns)),
                    max(4, len(correlations.index) * 0.8)
                )
            )

            image = ax.imshow(
                correlations,
                cmap="coolwarm",
                vmin=-1,
                vmax=1
            )

            fig.colorbar(
                image,
                ax=ax,
                label="Coefficient de corrélation"
            )

            ax.set_xticks(range(len(correlations.columns)))
            ax.set_xticklabels(
                correlations.columns,
                rotation=45,
                ha="right"
            )

            ax.set_yticks(range(len(correlations.index)))
            ax.set_yticklabels(correlations.index)

            for i in range(len(correlations.index)):
                for j in range(len(correlations.columns)):
                    valeur = correlations.iloc[i, j]

                    ax.text(
                        j,
                        i,
                        f"{valeur:.2f}",
                        ha="center",
                        va="center"
                    )

            ax.set_title("Matrice de corrélation")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.info(
                "Une corrélation mesure une association statistique. "
                "Elle ne prouve pas une relation de cause à effet."
            )

        st.divider()
        st.subheader("Détection des valeurs inhabituelles")

        anomalies = detecter_anomalies(df)

        if anomalies.empty:
            st.success(
                "Aucune valeur inhabituelle détectée par la méthode IQR."
            )
        else:
            st.warning(
                f"{len(anomalies)} valeur(s) inhabituelle(s) signalée(s)."
            )

            st.dataframe(
                anomalies,
                use_container_width=True
            )

        st.caption(
            "Une valeur signalée doit être examinée dans son contexte. "
            "Une anomalie statistique n'est pas forcément une erreur."
        )


# =========================================================
# 8. INTELLIGENCE ARTIFICIELLE
# =========================================================

elif page == "Intelligence artificielle":

    st.header("🧠 Intelligence artificielle")
    st.write(
        "Compare une régression linéaire et une forêt aléatoire "
        "à une méthode de référence."
    )

    if df is None:
        st.info("Importe un fichier CSV dans le panneau latéral.")

    else:
        numeriques = obtenir_numeriques(df)

        if numeriques.shape[1] < 2:
            st.warning(
                "Il faut au moins deux colonnes numériques "
                "pour entraîner un modèle."
            )

        else:
            candidates = [
                colonne for colonne in numeriques.columns
                if "temperature" in colonne.lower()
                or "température" in colonne.lower()
                or "temp" in colonne.lower()
            ]

            valeur_defaut = (
                list(numeriques.columns).index(candidates[0])
                if candidates else len(numeriques.columns) - 1
            )

            cible = st.selectbox(
                "Choisis la variable à prédire",
                list(numeriques.columns),
                index=valeur_defaut
            )

            st.caption(
                "Le modèle utilise les autres colonnes numériques "
                "comme variables explicatives."
            )

            if st.button(
                "Entraîner et comparer les modèles",
                type="primary"
            ):
                donnees_model = numeriques.dropna(
                    subset=[cible]
                ).copy()

                X = donnees_model.drop(columns=[cible])
                y = donnees_model[cible]

                X = X.dropna(axis=1, how="all")

                if len(y) < 10 or X.shape[1] == 0:
                    st.warning(
                        "Il faut au moins 10 observations exploitables "
                        "et une variable explicative."
                    )

                else:
                    try:
                        X_train, X_test, y_train, y_test = (
                            train_test_split(
                                X,
                                y,
                                test_size=0.25,
                                random_state=42
                            )
                        )

                        modeles = {
                            "Référence": make_pipeline(
                                SimpleImputer(strategy="median"),
                                DummyRegressor(strategy="mean")
                            ),
                            "Régression linéaire": make_pipeline(
                                SimpleImputer(strategy="median"),
                                LinearRegression()
                            ),
                            "Random Forest": make_pipeline(
                                SimpleImputer(strategy="median"),
                                RandomForestRegressor(
                                    n_estimators=100,
                                    min_samples_leaf=2,
                                    random_state=42
                                )
                            )
                        }

                        metriques = []
                        predictions_modeles = {}

                        with st.spinner("Entraînement des modèles..."):
                            for nom, modele in modeles.items():
                                modele.fit(X_train, y_train)

                                predictions = modele.predict(X_test)

                                erreur = mean_absolute_error(
                                    y_test,
                                    predictions
                                )

                                metriques.append({
                                    "Modèle": nom,
                                    "Erreur absolue moyenne": round(
                                        erreur, 4
                                    )
                                })

                                predictions_modeles[nom] = predictions

                        tableau_metriques = pd.DataFrame(metriques)

                        meilleur = min(
                            metriques,
                            key=lambda ligne: ligne[
                                "Erreur absolue moyenne"
                            ]
                        )["Modèle"]

                        prediction_meilleure = (
                            predictions_modeles[meilleur]
                        )

                        tableau_predictions = pd.DataFrame({
                            "Observation": y_test.index.astype(str),
                            "Valeur observée": y_test.to_numpy(),
                            "Prédiction du modèle retenu": (
                                prediction_meilleure
                            ),
                            "Erreur absolue": np.abs(
                                y_test.to_numpy()
                                - prediction_meilleure
                            )
                        })

                        st.session_state["resultat_ia"] = {
                            "signature": signature,
                            "cible": cible,
                            "metriques": metriques,
                            "predictions": tableau_predictions,
                            "meilleur": meilleur
                        }

                    except Exception as erreur:
                        st.error(
                            "L'entraînement n'a pas pu être terminé : "
                            f"{erreur}"
                        )

            resultat = st.session_state.get("resultat_ia")

            if (
                resultat is not None
                and resultat.get("signature") == signature
            ):
                st.subheader("Résultats de la comparaison")

                st.dataframe(
                    pd.DataFrame(resultat["metriques"]).sort_values(
                        "Erreur absolue moyenne"
                    ),
                    use_container_width=True
                )

                st.subheader("Prédictions sur les observations de test")

                st.dataframe(
                    resultat["predictions"].round(3),
                    use_container_width=True
                )

                st.info(
                    f"Erreur la plus faible sur ce test : "
                    f"{resultat['meilleur']}. "
                    "Cela ne prouve pas que ce modèle sera le meilleur "
                    "sur de nouvelles données."
                )

                st.caption(
                    "Les données doivent être suffisamment nombreuses "
                    "et représentatives pour évaluer la fiabilité "
                    "des prédictions."
                )


# =========================================================
# 9. RAPPORTS SCIENTIFIQUES
# =========================================================

elif page == "Rapports scientifiques":

    st.header("📄 Rapports scientifiques")

    if df is None:
        st.info(
            "Importe un fichier CSV pour générer un rapport."
        )

    else:
        rapport = generer_rapport(df, signature)

        st.write(
            "Le rapport rassemble les statistiques, les corrélations "
            "disponibles, les anomalies signalées et, si elle a été "
            "effectuée, la comparaison des modèles d'IA."
        )

        st.text_area(
            "Aperçu du rapport",
            rapport,
            height=400
        )

        st.download_button(
            "Télécharger le rapport scientifique",
            data=rapport,
            file_name="rapport_nexora_science.txt",
            mime="text/plain"
        )


# =========================================================
# 10. PIED DE PAGE
# =========================================================

st.markdown("""
<div class="footer">
    NEXORA SCIENCE · Plateforme expérimentale d'analyse scientifique
</div>
""", unsafe_allow_html=True)
