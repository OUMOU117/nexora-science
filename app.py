import streamlit as st
import pandas as pd

st.set_page_config(page_title="NEXORA SCIENCE", page_icon="🔬", layout="wide")

st.title("🔬 NEXORA SCIENCE")
st.subheader("Plateforme d'analyse scientifique")
st.write(
    "Importe un fichier CSV pour explorer tes données, créer des graphiques "
    "et générer un rapport scientifique."
)

fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

if fichier is not None:
    try:
        df = pd.read_csv(fichier)

        if df.empty:
            st.warning("Le fichier est vide.")
        else:
            st.success("Fichier importé avec succès !")

            st.subheader("Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)

            numeriques = df.select_dtypes(include="number")

            st.subheader("Résumé du jeu de données")
            col1, col2, col3 = st.columns(3)
            col1.metric("Nombre de lignes", len(df))
            col2.metric("Nombre de colonnes", len(df.columns))
            col3.metric(
                "Valeurs manquantes",
                int(df.isna().sum().sum())
            )

            if not numeriques.empty:
                st.subheader("Statistiques")
                st.dataframe(
                    numeriques.describe().round(2),
                    use_container_width=True
                )

                colonne = st.selectbox(
                    "Choisis une colonne numérique à visualiser",
                    numeriques.columns
                )

                st.subheader(f"Graphique : {colonne}")
                st.line_chart(numeriques[colonne].reset_index(drop=True))

            else:
                st.info("Aucune colonne numérique n'a été détectée.")

            rapport = (
                "NEXORA SCIENCE — RAPPORT SCIENTIFIQUE\n"
                "=====================================\n\n"
                f"Nombre de lignes : {len(df)}\n"
                f"Nombre de colonnes : {len(df.columns)}\n"
                f"Valeurs manquantes : {int(df.isna().sum().sum())}\n\n"
                "STATISTIQUES\n"
                + (
                    numeriques.describe().round(2).to_string()
                    if not numeriques.empty
                    else "Aucune colonne numérique détectée."
                )
            )

            st.subheader("Rapport scientifique")
            st.text_area("Rapport généré", rapport, height=250)

            st.download_button(
                "Télécharger le rapport",
                data=rapport,
                file_name="rapport_nexora_science.txt",
                mime="text/plain"
            )

    except Exception as erreur:
        st.error(f"Impossible d'analyser ce fichier : {erreur}")

else:
    st.info("Commence par importer un fichier CSV.")
