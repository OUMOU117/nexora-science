
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="NEXORA SCIENCE",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 NEXORA SCIENCE")
st.subheader("Plateforme d'analyse scientifique")

st.write(
    "Importe un fichier CSV pour explorer tes données, "
    "visualiser les variables, détecter des valeurs inhabituelles "
    "et générer un rapport scientifique."
)

fichier = st.file_uploader(
    "Importer un fichier CSV",
    type=["csv"]
)

if fichier is not None:
    try:
        df = pd.read_csv(fichier)

        if df.empty:
            st.warning("Le fichier est vide.")
        else:
            st.success("Fichier importé avec succès !")

            st.subheader("1. Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)

            numeriques = df.select_dtypes(include="number")

            st.subheader("2. Résumé du jeu de données")

            col1, col2, col3 = st.columns(3)
            col1.metric("Nombre de lignes", len(df))
            col2.metric("Nombre de colonnes", len(df.columns))
            col3.metric(
                "Valeurs manquantes",
                int(df.isna().sum().sum())
            )

            rapport = [
                "NEXORA SCIENCE — RAPPORT SCIENTIFIQUE",
                "=" * 42,
                "",
                f"Nombre de lignes : {len(df)}",
                f"Nombre de colonnes : {len(df.columns)}",
                f"Valeurs manquantes : {int(df.isna().sum().sum())}",
                "",
            ]

            if not numeriques.empty:
                st.subheader("3. Statistiques descriptives")

                statistiques = numeriques.describe().round(2)
                st.dataframe(statistiques, use_container_width=True)

                rapport.extend([
                    "STATISTIQUES DESCRIPTIVES",
                    statistiques.to_string(),
                    ""
                ])

                st.subheader("4. Visualisation")

                colonne = st.selectbox(
                    "Choisis une variable numérique",
                    numeriques.columns
                )

                st.line_chart(
                    numeriques[colonne].reset_index(drop=True)
                )

                st.subheader("5. Détection des valeurs inhabituelles")

                st.write(
                    "La méthode utilisée est celle de l'intervalle "
                    "interquartile (IQR). Elle signale les valeurs "
                    "situées en dehors des limites statistiques."
                )

                anomalies_par_variable = {}
                details_anomalies = []

                for nom_colonne in numeriques.columns:
                    serie = numeriques[nom_colonne].dropna()

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
                        (numeriques[nom_colonne] < limite_basse)
                        | (numeriques[nom_colonne] > limite_haute)
                    )

                    valeurs = numeriques.loc[masque, nom_colonne]

                    if not valeurs.empty:
                        anomalies_par_variable[nom_colonne] = len(valeurs)

                        for index, valeur in valeurs.items():
                            details_anomalies.append({
                                "Ligne du fichier": index + 2,
                                "Variable": nom_colonne,
                                "Valeur inhabituelle": valeur,
                                "Limite basse": round(limite_basse, 2),
                                "Limite haute": round(limite_haute, 2)
                            })

                if details_anomalies:
                    st.warning(
                        f"{len(details_anomalies)} valeur(s) inhabituelle(s) "
                        "ont été détectées."
                    )

                    st.dataframe(
                        pd.DataFrame(details_anomalies),
                        use_container_width=True
                    )

                    st.info(
                        "Une valeur inhabituelle n'est pas nécessairement "
                        "une erreur. Vérifie le contexte scientifique "
                        "avant de tirer une conclusion."
                    )

                    rapport.extend([
                        "VALEURS INHABITUELLES",
                        f"Nombre de valeurs signalées : {len(details_anomalies)}",
                        pd.DataFrame(details_anomalies).to_string(index=False),
                        ""
                    ])
                else:
                    st.success(
                        "Aucune valeur inhabituelle détectée par cette méthode."
                    )

                    rapport.extend([
                        "VALEURS INHABITUELLES",
                        "Aucune valeur inhabituelle détectée par la méthode IQR.",
                        ""
                    ])

            else:
                st.info(
                    "Aucune colonne numérique n'a été détectée. "
                    "Les statistiques et la détection des anomalies "
                    "ne sont pas disponibles."
                )

                rapport.extend([
                    "STATISTIQUES",
                    "Aucune colonne numérique détectée.",
                    ""
                ])

            st.subheader("6. Rapport scientifique")

            texte_rapport = "\n".join(rapport)

            st.text_area(
                "Rapport généré",
                texte_rapport,
                height=300
            )

            st.download_button(
                "Télécharger le rapport scientifique",
                data=texte_rapport,
                file_name="rapport_nexora_science.txt",
                mime="text/plain"
            )

    except Exception as erreur:
        st.error(
            f"Impossible d'analyser ce fichier. "
            f"Vérifie son format CSV. Détail : {erreur}"
        )

else:
    st.info("Commence par importer un fichier CSV.")
```
