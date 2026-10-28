

```python
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
    "Importe un fichier CSV pour analyser tes données, "
    "visualiser les résultats et générer un rapport scientifique."
)

fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

if fichier is None:
    st.info("Commence par importer un fichier CSV.")
else:
    try:
        df = pd.read_csv(fichier)

        if df.empty:
            st.warning("Le fichier est vide.")
        else:
            st.success("Fichier importé avec succès !")

            st.subheader("1. Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)

            numeriques = df.select_dtypes(include="number")
            valeurs_manquantes = int(df.isna().sum().sum())

            st.subheader("2. Résumé des données")

            col1, col2, col3 = st.columns(3)
            col1.metric("Nombre de lignes", len(df))
            col2.metric("Nombre de colonnes", len(df.columns))
            col3.metric("Valeurs manquantes", valeurs_manquantes)

            rapport = [
                "NEXORA SCIENCE - RAPPORT SCIENTIFIQUE",
                "=" * 42,
                "",
                f"Nombre de lignes : {len(df)}",
                f"Nombre de colonnes : {len(df.columns)}",
                f"Valeurs manquantes : {valeurs_manquantes}",
                ""
            ]

            anomalies = []

            if not numeriques.empty:
                st.subheader("3. Statistiques descriptives")

                statistiques = numeriques.describe().round(2)
                st.dataframe(statistiques, use_container_width=True)

                rapport.extend([
                    "STATISTIQUES DESCRIPTIVES",
                    statistiques.to_string(),
                    ""
                ])

                st.subheader("4. Graphique")

                colonne = st.selectbox(
                    "Choisis une variable numérique",
                    list(numeriques.columns)
                )

                st.line_chart(numeriques[colonne].reset_index(drop=True))

                st.subheader("5. Détection des valeurs inhabituelles")

                st.write(
                    "La méthode IQR repère les valeurs situées en dehors "
                    "de limites calculées à partir des quartiles."
                )

                for nom in numeriques.columns:
                    serie = numeriques[nom].dropna()

                    if len(serie) < 4:
                        continue

                    q1 = serie.quantile(0.25)
                    q3 = serie.quantile(0.75)
                    iqr = q3 - q1

                    if iqr == 0:
                        continue

                    borne_basse = q1 - 1.5 * iqr
                    borne_haute = q3 + 1.5 * iqr

                    masque = (
                        (numeriques[nom] < borne_basse)
                        | (numeriques[nom] > borne_haute)
                    )

                    for index, valeur in numeriques.loc[masque, nom].items():
                        anomalies.append({
                            "Ligne du fichier": index + 2,
                            "Variable": nom,
                            "Valeur": valeur,
                            "Limite basse": round(borne_basse, 2),
                            "Limite haute": round(borne_haute, 2)
                        })

                if anomalies:
                    tableau_anomalies = pd.DataFrame(anomalies)

                    st.warning(
                        f"{len(anomalies)} valeur(s) inhabituelle(s) détectée(s)."
                    )
                    st.dataframe(
                        tableau_anomalies,
                        use_container_width=True
                    )

                    rapport.extend([
                        "VALEURS INHABITUELLES",
                        tableau_anomalies.to_string(index=False),
                        ""
                    ])
                else:
                    st.success(
                        "Aucune valeur inhabituelle détectée par la méthode IQR."
                    )
                    rapport.extend([
                        "VALEURS INHABITUELLES",
                        "Aucune valeur détectée par la méthode IQR.",
                        ""
                    ])

                st.info(
                    "Une valeur inhabituelle n'est pas forcément une erreur. "
                    "Elle doit être vérifiée dans son contexte scientifique."
                )

            else:
                st.info("Aucune colonne numérique n'a été détectée.")
                rapport.extend([
                    "STATISTIQUES",
                    "Aucune colonne numérique détectée.",
                    ""
                ])

            texte_rapport = "\n".join(rapport)

            st.subheader("6. Rapport scientifique")
            st.text_area(
                "Rapport généré",
                texte_rapport,
                height=300
            )

            st.download_button(
                "Télécharger le rapport",
                data=texte_rapport,
                file_name="rapport_nexora_science.txt",
                mime="text/plain"
            )

    except Exception as erreur:
        st.error(f"Erreur lors de l'analyse du CSV : {erreur}")
```
