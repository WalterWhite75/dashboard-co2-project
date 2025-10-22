import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, ctx
import dash_bootstrap_components as dbc
from pathlib import Path
import numpy as np

#  1) CHARGEMENT DES DONNÉES 
path = Path(__file__).resolve().parents[1] / "data" / "DonnéesC02_clean.csv"
if not path.exists():
    raise FileNotFoundError(f"Fichier introuvable : {path}")

df = pd.read_csv(path)

# FILTRAGE : ne garder que les vrais pays (codes ISO à 3 lettres) et exclure les agrégats/regions
a_exclure = [
    "World", "IDA & IBRD total", "IBRD only", "IDA total", "Low & middle income", "High income",
    "Middle income", "Upper middle income", "Lower middle income", "OECD members", "Non-OECD members",
    "Europe & Central Asia", "Sub-Saharan Africa", "East Asia & Pacific", "Latin America & Caribbean",
    "North America", "South Asia", "Arab World", "European Union", "Euro area", "Small states",
    "Fragile and conflict affected situations", "Least developed countries: UN classification",
    "Early-demographic dividend", "Late-demographic dividend", "Post-demographic dividend",
    "East Asia & Pacific (IDA & IBRD countries)", "Europe & Central Asia (IDA & IBRD countries)",
    "South Asia (IDA & IBRD)", "Middle East & North Africa (IDA & IBRD countries)",
    "East Asia & Pacific (excluding high income)"
]

# Garder uniquement les lignes avec code ISO-3 et exclure les regroupements
_df = df.copy()
_df = _df[_df["Country Code"].str.match(r"^[A-Z]{3}$", na=False)]
_df = _df[~_df["Country Name"].isin(a_exclure)].copy()
# Exclure aussi par codes agrégés et motifs de noms
codes_aggregats = {
    "WLD","EAP","ECA","LCN","MEA","MNA","NAC","SAS","SSA",
    "EUU","EMU","OED","HIC","UMC","LMC","LIC","MIC","IBD","IBT","IDA"
}
_df = _df[~_df["Country Code"].isin(codes_aggregats)]
# Exclure quelques motifs explicites (zones, regroupements)
patterns = [r"\\bincome\\b", r"countries", r"members", r"dividend", r"Euro area", r"European Union", r"Arab World", r"IDA", r"IBRD"]
mask_pat = _df["Country Name"].str.contains("|".join(patterns), case=False, na=False)
# Certains agrégats contiennent un '&' (ex: "Afghanistan & Pakistan"). On enlève ceux-là aussi.
mask_amp = _df["Country Name"].str.contains("&", na=False)
_df = _df[~(mask_pat | mask_amp)]

# Colonnes d'années et conversion numérique
annees = [col for col in _df.columns if str(col).isdigit()]
_df[annees] = _df[annees].apply(pd.to_numeric, errors="coerce")

# Remplace df de travail par df_pays pour la suite du script
df_pays = _df
nb_pays = df_pays["Country Name"].nunique()
nb_annees = len(annees)

# Fonction utilitaire pour construire la carte pour une année donnée
def build_carte(annee: str):
    title = f"Émissions de CO₂ par pays en {annee} (millions de tonnes)"
    fig = px.choropleth(
        df_pays,
        locations="Country Code",
        color=annee,
        hover_name="Country Name",
        color_continuous_scale="Viridis",
        title=title,
        labels={annee: "Émissions (millions de tonnes)"}
    )
    fig.update_layout(template="plotly_white", height=800, margin=dict(l=0, r=0, t=60, b=0))
    fig.update_geos(
        showcountries=True,
        countrycolor="lightgray",
        showcoastlines=True,
        coastlinecolor="white",
        showland=True,
        landcolor="#f9f9f9",
        projection_type="natural earth",
        bgcolor="rgba(240,240,240,0)"
    )
    fig.update_coloraxes(
        colorbar_title="Émissions (MtCO₂)",
        colorbar_thickness=18,
        colorbar_tickfont_size=11,
        colorbar_title_side="right"
    )
    return fig

# Fonction utilitaire pour construire le Top 10 des pays pour une année donnée
def build_top10(annee: str):
    emissions = df_pays[["Country Name", annee]].dropna()
    emissions = emissions.sort_values(annee, ascending=False).head(10)
    emissions["Émissions (millions de tonnes)"] = emissions[annee]
    emissions_sorted = emissions.sort_values("Émissions (millions de tonnes)", ascending=True)

    fig = px.bar(
        emissions_sorted,
        y="Country Name",
        x="Émissions (millions de tonnes)",
        orientation="h",
        title=f"Top 10 des pays avec les émissions de CO₂ les plus élevées en {annee} (millions de tonnes)",
        text=emissions_sorted["Émissions (millions de tonnes)"].round(0).astype(int).astype(str) + " M",
        color="Émissions (millions de tonnes)",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=220, r=40, t=70, b=50),
        xaxis_title="Émissions (millions de tonnes)",
        yaxis_title="Pays",
        title_font=dict(size=13),
        xaxis_title_font=dict(size=12),
        yaxis_title_font=dict(size=12),
        xaxis_tickfont=dict(size=11),
        yaxis_tickfont=dict(size=11),
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10))
    return fig

# 3) ANALYSE DES ÉMISSIONS DE CO2

# Calcul des émissions totales par année (somme mondiale)
emissions_par_annee = df_pays[annees].sum().reset_index()
emissions_par_annee.columns = ["Année", "Émissions totales (millions de tonnes)"]
emissions_par_annee["Année"] = emissions_par_annee["Année"].astype(str)

# CAGR (Compound Annual Growth Rate) entre 1990 et 2023
emission_1990 = emissions_par_annee.loc[emissions_par_annee["Année"] == "1990", "Émissions totales (millions de tonnes)"].values[0]
emission_2023 = emissions_par_annee.loc[emissions_par_annee["Année"] == "2023", "Émissions totales (millions de tonnes)"].values[0]
years_diff = 2023 - 1990
cagr = ((emission_2023 / emission_1990) ** (1 / years_diff) - 1) * 100

# Année du pic d’émission
annee_pic = emissions_par_annee.loc[emissions_par_annee["Émissions totales (millions de tonnes)"].idxmax(), "Année"]

# % de pays ayant réduit leurs émissions depuis 1990
emission_1990_pays = df_pays[["Country Name", "1990"]].dropna()
emission_2023_pays = df_pays[["Country Name", "2023"]].dropna()
emission_compare = pd.merge(emission_1990_pays, emission_2023_pays, on="Country Name", suffixes=("_1990", "_2023"))
emission_compare["reduction"] = emission_compare["2023"] < emission_compare["1990"]
pct_reduction = (emission_compare["reduction"].sum() / len(emission_compare)) * 100

# Calcul des émissions totales par pays pour 2023
emissions_2023 = df_pays[["Country Name", "2023"]].dropna()
emissions_2023 = emissions_2023.sort_values("2023", ascending=False).head(10)
emissions_2023["Émissions (millions de tonnes)"] = emissions_2023["2023"]

# Part des 10 pays les plus émetteurs en 2023
total_2023 = df_pays["2023"].sum()
part_top10 = emissions_2023["2023"].sum() / total_2023 * 100

# Ratio 80/20: nombre de pays cumulant 80% des émissions mondiales
df_sorted = df_pays[["Country Name", "2023"]].dropna().sort_values("2023", ascending=False)
df_sorted["cumulative"] = df_sorted["2023"].cumsum()
threshold_80 = 0.8 * total_2023
num_pays_80 = df_sorted[df_sorted["cumulative"] <= threshold_80].shape[0]
ratio_80_20 = num_pays_80 / nb_pays * 100

# Score environnemental global : moyenne des réductions de chaque pays depuis 1990 (en %)
emission_compare["reduction_pct"] = ((emission_compare["1990"] - emission_compare["2023"]) / emission_compare["1990"]) * 100
score_env = emission_compare["reduction_pct"].mean()

# Graphique principal : évolution globale des émissions
fig_evolution = px.line(
    emissions_par_annee,
    x="Année",
    y="Émissions totales (millions de tonnes)",
    title="Évolution globale des émissions de CO₂ (millions de tonnes) (1970–2023)",
    markers=True
)
fig_evolution.update_traces(line_color="blue", marker_size=7)
fig_evolution.update_layout(template="plotly_white")
fig_evolution.update_layout(
    title_font=dict(size=13),
    xaxis_title_font=dict(size=12),
    yaxis_title_font=dict(size=12),
    xaxis_tickfont=dict(size=11),
    yaxis_tickfont=dict(size=11),
)

# Préparer figure par défaut et options d'années pour le sélecteur
annee_defaut = "2023" if "2023" in annees else annees[-1]
fig_carte_default = build_carte(annee_defaut)
annee_options = [{"label": a, "value": a} for a in annees]

# Graphique secondaire : top 10 des pays avec les émissions les plus élevées en 2023 (barres horizontales)
emissions_2023_sorted = emissions_2023.sort_values("Émissions (millions de tonnes)", ascending=True)
fig_top = px.bar(
    emissions_2023_sorted,
    y="Country Name",
    x="Émissions (millions de tonnes)",
    orientation="h",
    title="Top 10 des pays avec les émissions de CO₂ les plus élevées en 2023 (millions de tonnes)",
    text=emissions_2023_sorted["Émissions (millions de tonnes)"].round(0).astype(int).astype(str) + " M",
    color="Émissions (millions de tonnes)",
    color_continuous_scale="Blues"
)
fig_top.update_layout(
    template="plotly_white",
    height=650,
    margin=dict(l=220, r=40, t=70, b=50),
    xaxis_title="Émissions (millions de tonnes)",
    yaxis_title="Pays"
)
fig_top.update_layout(
    title_font=dict(size=13),
    xaxis_title_font=dict(size=12),
    yaxis_title_font=dict(size=12),
    xaxis_tickfont=dict(size=11),
    yaxis_tickfont=dict(size=11),
)
fig_top.update_traces(textposition="outside", textfont=dict(size=10))

# Graphique cumulative 80/20 pour visualiser la concentration mondiale
df_sorted["cumulative_pct"] = df_sorted["cumulative"] / total_2023 * 100
df_sorted["rank_pct"] = np.arange(1, len(df_sorted) + 1) / len(df_sorted) * 100
fig_8020 = px.line(
    df_sorted,
    x="rank_pct",
    y="cumulative_pct",
    title="Courbe cumulative 80/20 des émissions mondiales de CO₂ (millions de tonnes)",
    labels={"rank_pct": "Pourcentage de pays (%)", "cumulative_pct": "Pourcentage des émissions cumulées (%)"}
)
fig_8020.add_scatter(x=[20, 100], y=[80, 100], mode="lines", name="Repère 80/20", line=dict(dash="dash", color="red"))
fig_8020.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=0.98))
fig_8020.update_layout(template="plotly_white")
fig_8020.update_layout(
    title_font=dict(size=13),
    xaxis_title_font=dict(size=12),
    yaxis_title_font=dict(size=12),
    xaxis_tickfont=dict(size=11),
    yaxis_tickfont=dict(size=11),
)

#  ESTIMATIONS / PROJECTIONS FUTURES 
horizon = 2040
annees_existantes = emissions_par_annee["Année"].astype(int)
emissions_existantes = emissions_par_annee["Émissions totales (millions de tonnes)"]

annees_projection = np.arange(2024, horizon + 1)
projections = emission_2023 * ((1 + (cagr / 100)) ** (annees_projection - 2023))

df_projection = pd.DataFrame({
    "Année": np.concatenate([annees_existantes, annees_projection]),
    "Émissions (millions de tonnes)": np.concatenate([emissions_existantes, projections])
})
df_projection["Type"] = ["Historique"] * len(annees_existantes) + ["Projection"] * len(annees_projection)

# Courbe de projection avec zone d’incertitude ±5%
df_projection["Min_estimation"] = df_projection["Émissions (millions de tonnes)"] * 0.95
df_projection["Max_estimation"] = df_projection["Émissions (millions de tonnes)"] * 1.05

fig_projection = px.line(
    df_projection,
    x="Année",
    y="Émissions (millions de tonnes)",
    color="Type",
    title="Projection exponentielle des émissions mondiales de CO₂ (jusqu’à 2040)",
    color_discrete_map={"Historique": "blue", "Projection": "orange"}
)
fig_projection.add_traces(px.scatter(df_projection[df_projection["Type"] == "Projection"], x="Année", y="Émissions (millions de tonnes)").data)
fig_projection.add_scatter(
    x=np.concatenate([df_projection["Année"], df_projection["Année"][::-1]]),
    y=np.concatenate([df_projection["Max_estimation"], df_projection["Min_estimation"][::-1]]),
    fill="toself",
    fillcolor="rgba(255,165,0,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    showlegend=False,
    name="Intervalle ±5%"
)
fig_projection.update_layout(
    template="plotly_white",
    title_font=dict(size=13),
    xaxis_title_font=dict(size=12),
    yaxis_title_font=dict(size=12),
    xaxis_tickfont=dict(size=11),
    yaxis_tickfont=dict(size=11),
)

#  PROJECTIONS PAR PAYS (corrigé pour éviter les DataFrames vides) 
df_cagr_pays = df_pays[["Country Name", "1990", "2023"]].copy()

# Remplacer les valeurs manquantes de 1990 par la moyenne mondiale
mean_1990 = df_cagr_pays["1990"].mean(skipna=True)
df_cagr_pays["1990"] = df_cagr_pays["1990"].fillna(mean_1990)

# Supprimer les lignes avec valeurs nulles ou négatives
df_cagr_pays = df_cagr_pays[(df_cagr_pays["1990"] > 0) & (df_cagr_pays["2023"] > 0)]

# Calcul du CAGR (taux de croissance annuel composé)
df_cagr_pays["CAGR"] = ((df_cagr_pays["2023"] / df_cagr_pays["1990"]) ** (1 / (2023 - 1990)) - 1)

# Filtrer les valeurs aberrantes (croissance > ±20 % par an)
df_cagr_pays = df_cagr_pays[(df_cagr_pays["CAGR"] < 0.20) & (df_cagr_pays["CAGR"] > -0.20)]

# Calculer les projections pour 2030, 2040 et 2050
for horizon_future in [2030, 2040, 2050]:
    df_cagr_pays[f"Projection_{horizon_future}"] = df_cagr_pays["2023"] * ((1 + df_cagr_pays["CAGR"]) ** (horizon_future - 2023))

# Vérification de la complétude
if df_cagr_pays.empty:
    print("⚠️ Aucune donnée valide pour les projections par pays. Vérifiez les colonnes 1990 et 2023.")

# KPI : total mondial d'émissions en 2023 (en millions de tonnes)
total_mondial_2023 = total_2023.round(2)
# Format clair pour affichage (séparateur espace) + unité
total_mondial_2023_fmt = f"{total_mondial_2023:,.0f}".replace(",", " ") + " MtCO₂"

# Variation % des émissions entre 1990 et 2023
variation_pct = ((emission_2023 - emission_1990) / emission_1990) * 100

#  5) CONSTRUCTION DU DASHBOARD 
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Analyse des émissions de CO₂ — Thème clair"

#  Fonctions d'insights (à placer juste après la section KPI, avant le layout)
def generate_insights_global():
    variation_text = f"Depuis 1990, les émissions mondiales de CO₂ ont {'augmenté' if variation_pct > 0 else 'diminué'} de {abs(variation_pct):.1f} %."
    total_text = f"En 2023, elles atteignent {total_mondial_2023_fmt}."
    ratio_text = f"Environ 20 % des pays concentrent {ratio_80_20:.1f} % des émissions totales, ce qui souligne une forte concentration géographique des émissions."
    return f"{variation_text} {total_text} {ratio_text}"

def generate_insights_pays(annee):
    df_annee = df_pays[["Country Name", annee]].dropna().sort_values(annee, ascending=False)
    total_annee = df_annee[annee].sum()
    top10 = df_annee.head(10)
    part_top10 = top10[annee].sum() / total_annee * 100
    if len(top10) >= 2:
        p1, v1 = top10.iloc[0]["Country Name"], top10.iloc[0][annee] / total_annee * 100
        p2, v2 = top10.iloc[1]["Country Name"], top10.iloc[1][annee] / total_annee * 100
        text = (f"En {annee}, le pays le plus émetteur est {p1} ({v1:.1f} %), suivi de {p2} ({v2:.1f} %). "
                f"Ensemble, les 10 premiers pays représentent environ {part_top10:.1f} % des émissions mondiales.")
    else:
        text = f"Données insuffisantes pour l'année {annee}."
    return text

navbar = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H3("Analyse mondiale du CO₂", className="mb-0", 
                    style={"color": "white", "fontWeight": "700", "fontSize": "1.5rem"}),
            html.Small("Tableau de bord interactif (1970–2023)", 
                       style={"color": "#dfe6f0", "fontSize": "0.9rem"})
        ], className="d-flex flex-column justify-content-center align-items-start")
    ]),
    color="#003366",
    dark=True,
    class_name="shadow-sm",
    style={"height": "80px", "paddingLeft": "30px", "borderBottom": "3px solid #0056b3"}
)

header = dbc.Container([
    html.Div([
        html.H1(
            "Tableau de bord des émissions mondiales de CO₂",
            className="display-6",
            style={"color": "#0056b3", "marginBottom": "0.5rem"}
        ),
        html.P(
            "Analyse interactive des émissions (1970–2023) — Source: Banque mondiale",
            style={"color": "#495057", "marginBottom": "0"}
        ),
    ], className="text-center my-3")
])

app.layout = dbc.Container([
    navbar,
    header,
    html.H3("Tableau de bord — Analyse des émissions réelles de CO₂ (1970–2023)", className="text-center my-3"),

    # Ligne 1 : KPI globaux (Total, Variation %, CAGR)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Émissions totales mondiales en 2023 (MtCO₂)", className="card-title"),
                html.H4(total_mondial_2023_fmt, className="fw-bold", style={"fontSize": "1.1rem"})
            ])
        ], color="light", inverse=False, style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)", "minHeight": "130px"}), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Variation des émissions depuis 1990", className="card-title"),
                html.H4(f"{variation_pct:.2f} %", className="text-danger" if variation_pct > 0 else "text-success", style={"fontSize": "1.1rem"})
            ])
        ], color="light", inverse=False, style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)", "minHeight": "130px"}), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("CAGR (1990-2023)", className="card-title"),
                html.H4(f"{cagr:.2f} %", className="text-danger" if cagr > 0 else "text-success", style={"fontSize": "1.1rem"})
            ])
        ], color="light", inverse=False, style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)", "minHeight": "130px"}), width=4),
    ], className="mb-4"),

    # Ligne 2 : KPI d’impact (Pic d’émission, % pays en baisse)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Année du pic d’émission", className="card-title"),
                html.H4(annee_pic, className="text-primary", style={"fontSize": "1.1rem"})
            ])
        ], color="light", inverse=False, style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)", "minHeight": "130px"}), width=6),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("% de pays ayant réduit leurs émissions depuis 1990", className="card-title"),
                html.H4(f"{pct_reduction:.2f} %", className="text-success", style={"fontSize": "1.1rem"})
            ])
        ], color="light", inverse=False, style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)", "minHeight": "130px"}), width=6),
    ], className="mb-4"),

# Bouton d'export CSV des indicateurs
    dbc.Row([
        dbc.Col(html.Div([
            dbc.Button(" Exporter les indicateurs", id="btn-export-csv", color="primary", className="me-2"),
            dcc.Download(id="download-kpi-csv")
        ], className="text-center mt-2"), width=12)
    ], className="mb-4"),

    html.Hr(style={"marginTop": "20px", "marginBottom": "20px"}),

    # Onglets : 1) À propos, 2) Tendances globales, 3) Pays & carte, 4) Estimations / Projections
    dcc.Tabs([
        dcc.Tab(label="À propos", children=[
            dbc.Container([
                html.Br(),
                html.H4("À propos du projet", className="text-primary fw-bold text-center mb-3"),
                html.P(
                    "Ce tableau de bord a été développé dans le cadre du Master 2 Big Data, Analyse et Business Intelligence à l’Université Sorbonne Paris Nord.",
                    className="text-center", style={"maxWidth": "80%", "margin": "auto", "fontSize": "1.05rem"}
                ),
                html.P(
                    "L’objectif du projet est d’analyser l’évolution mondiale des émissions de CO₂ et de mettre en évidence les disparités entre les pays grâce à un tableau de bord interactif réalisé avec Dash, Plotly et Pandas.",
                    className="text-center", style={"maxWidth": "85%", "margin": "auto", "fontSize": "1.0rem"}
                ),
                html.Hr(),
                html.H5(" Source des données :", className="fw-bold mt-4"),
                html.P("Banque mondiale — Indicateur : Carbon dioxide (CO₂) emissions from Industrial Processes (EN.GHG.CO2.IC.MT.CE.AR5)."),
                html.H5(" Méthodologie :", className="fw-bold mt-4"),
                html.Ul([
                    html.Li("Nettoyage et filtrage des agrégats régionaux et des données non pays."),
                    html.Li("Calcul des indicateurs clés : CAGR, ratio 80/20, part du Top 10, etc."),
                    html.Li("Création d’un ETL simple et d’un tableau de bord multi-onglets."),
                    html.Li("Ajout d’insights automatiques et d’un export des indicateurs en CSV.")
                ]),
                html.H5(" Auteur :", className="fw-bold mt-4"),
                html.P("Projet réalisé par Mevlut Cakin — Étudiant en Master 2 Big Data, Analyse et Business Intelligence (Université Sorbonne Paris Nord)."),
                html.H5(" GitHub :", className="fw-bold mt-4"),
                html.A("Lien vers le dépôt GitHub", href="https://github.com/", target="_blank", style={"color": "#0d6efd"}),
                html.Br(), html.Br()
            ])
        ]),
        dcc.Tab(label="Tendances globales", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_evolution, style={"height": "550px"}), width=6),
                dbc.Col(dcc.Graph(figure=fig_8020, style={"height": "550px"}), width=6),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        generate_insights_global(),
                        id="insight-global",
                        className="text-muted fst-italic text-center mt-3 p-3 border rounded",
                        style={
                            "backgroundColor": "#f8f9fa",
                            "border": "1px solid #dee2e6",
                            "color": "#212529",
                            "maxWidth": "90%",
                            "margin": "auto",
                            "boxShadow": "0 0 10px rgba(0,0,0,0.05)",
                            "fontSize": "0.95rem",
                            "lineHeight": "1.5"
                        }
                    ),
                    width=12
                )
            ])
        ]),
        dcc.Tab(label="Pays & carte", children=[
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label("Année :", className="fw-bold me-2"),
                    dcc.Dropdown(
                        id="dropdown-annee",
                        options=annee_options,
                        value=annee_defaut,
                        clearable=False,
                        style={"width": "220px"}
                    ),
                    html.Label("Indicateur :", className="fw-bold ms-4 me-2"),
                    dcc.Dropdown(
                        id="dropdown-indicateur",
                        options=[
                            {"label": "Émissions totales", "value": "total"},
                            {"label": "Émissions par habitant", "value": "per_capita"},
                            {"label": "Variation depuis 1990", "value": "variation"}
                        ],
                        value="total",
                        clearable=False,
                        style={"width": "280px"}
                    )
                ], className="d-flex align-items-center mb-2"), width=12),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="carte-emissions", figure=fig_carte_default, style={"height": "800px"}), width=12),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="graph-top10", figure=build_top10(annee_defaut), style={"height": "650px"}), width=12),
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        id="insight-pays",
                        className="text-muted fst-italic text-center mt-3 p-3 border rounded",
                        style={
                            "backgroundColor": "#f8f9fa",
                            "border": "1px solid #dee2e6",
                            "color": "#212529",
                            "maxWidth": "90%",
                            "margin": "auto",
                            "boxShadow": "0 0 10px rgba(0,0,0,0.05)",
                            "fontSize": "0.95rem",
                            "lineHeight": "1.5"
                        }
                    ),
                    width=12
                )
            ])
        ]),
        dcc.Tab(label="Estimations / Projections", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_projection, style={"height": "600px"}), width=12),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            "Les projections reposent sur un modèle exponentiel basé sur le taux de croissance annuel composé (CAGR) observé entre 1990 et 2023. "
                            "Elles fournissent une estimation théorique si la tendance actuelle se maintient, sans changement majeur des politiques environnementales.",
                            className="text-muted fst-italic text-center p-3 border rounded",
                            style={
                                "backgroundColor": "#f8f9fa",
                                "border": "1px solid #dee2e6",
                                "color": "#212529",
                                "maxWidth": "90%",
                                "margin": "auto",
                                "boxShadow": "0 0 10px rgba(0,0,0,0.05)",
                                "fontSize": "0.95rem",
                                "lineHeight": "1.5"
                            }
                        ),
                        width=12
                    )
                ]),
                #  Ajout du Top 10 projection dynamique et insight automatique 
                dbc.Row([
                    dbc.Col([
                        html.Label("Horizon :", className="fw-bold me-2"),
                        dcc.Dropdown(
                            id="dropdown-horizon",
                            options=[
                                {"label": "2030", "value": 2030},
                                {"label": "2040", "value": 2040},
                                {"label": "2050", "value": 2050}
                            ],
                            value=2040,
                            clearable=False,
                            style={"width": "200px", "margin": "0 auto"}
                        )
                    ], width=12, className="text-center mb-3")
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="graph-top-projection", style={"height": "650px"}), width=12)
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            id="insight-projection",
                            className="text-muted fst-italic text-center mt-3 p-3 border rounded",
                            style={
                                "backgroundColor": "#f8f9fa",
                                "border": "1px solid #dee2e6",
                                "color": "#212529",
                                "maxWidth": "90%",
                                "margin": "auto",
                                "boxShadow": "0 0 10px rgba(0,0,0,0.05)",
                                "fontSize": "0.95rem",
                                "lineHeight": "1.5"
                            }
                        ),
                        width=12
                    )
                ]),
            ])
        ]),
        dcc.Tab(label="Facteurs explicatifs & géopolitique", children=[
            dbc.Container([
                html.Br(),
                #  Introduction 
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Comprendre les dynamiques globales", className="mb-2", style={"color": "#0056b3"}),
                                html.P(
                                    "Les émissions de CO₂ résultent d'une combinaison de facteurs économiques, énergétiques et géopolitiques."
                                    " Cette section met en contexte l'évolution observée (1970–2023) et les projections (2030–2050).",
                                    className="mb-0"
                                ),
                            ],
                            className="p-3 border rounded",
                            style={
                                "backgroundColor": "#eef4fb",
                                "border": "1px solid #dee2e6",
                                "boxShadow": "0 0 10px rgba(0,0,0,0.05)",
                            }
                        ), width=12
                    )
                ], className="mb-4"),

                #  Facteurs économiques & énergétiques 
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Croissance des pays émergents", className="text-primary"),
                            html.P(
                                "Depuis les années 1990, l'industrialisation rapide de la Chine, de l'Inde et de l'Asie du Sud-Est a fortement augmenté la demande"
                                " d'énergie et d'acier, tirant les émissions mondiales à la hausse.")
                        ])
                    ], color="light", style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "height": "100%"}), width=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Dépendance aux énergies fossiles", className="text-primary"),
                            html.P(
                                "Le charbon reste central dans la production électrique en Asie; le pétrole domine encore les transports; le gaz a progressé pour le chauffage"
                                " et l'industrie. La croissance de la demande énergétique mondiale compense en partie les gains d'efficacité.")
                        ])
                    ], color="light", style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "height": "100%"}), width=4),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Stabilisation relative des pays développés", className="text-primary"),
                            html.P(
                                "En Europe et en Amérique du Nord, les émissions ont ralenti grâce aux politiques climatiques, à la désindustrialisation partielle et"
                                " à l'essor des renouvelables; mais le transport et certains usages industriels restent intensifs en carbone.")
                        ])
                    ], color="light", style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "height": "100%"}), width=4),
                ], className="mb-4"),

                #  Timeline géopolitique (événements majeurs) 
                dbc.Row([
                    dbc.Col(html.H5("Événements géopolitiques et énergétiques marquants", className="mb-3", style={"color": "#0056b3"}), width=12)
                ]),
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            [
                                # Timeline container
                                html.Div([
                                    # 1973 choc pétrolier
                                    html.Div([
                                        html.Strong("1973"),
                                        html.P("1er choc pétrolier : flambée des prix, efficacité énergétique en Occident, bascule vers le charbon ailleurs.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                    # 2001 OMC
                                    html.Div([
                                        html.Strong("2001"),
                                        html.P("Chine à l'OMC : accélération de l'industrialisation et de l'empreinte carbone exportée.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                    # 2008-09 crise
                                    html.Div([
                                        html.Strong("2008–2009"),
                                        html.P("Crise financière mondiale : baisse temporaire des émissions, reprise rapide ensuite.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                    # 2015 Accord de Paris
                                    html.Div([
                                        html.Strong("2015"),
                                        html.P("Accord de Paris (COP21) : objectifs climatiques globaux, mise en place de politiques nationales.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                    # 2016 élection US
                                    html.Div([
                                        html.Strong("2016"),
                                        html.P("Élection de Donald Trump (USA) : relâchement de certaines régulations, retrait de l'Accord de Paris (annoncé 2017)."
                                               " Politiques de relocalisation industrielle pouvant accroître temporairement les émissions domestiques selon le mix énergétique.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                    # 2020 COVID
                                    html.Div([
                                        html.Strong("2020"),
                                        html.P("COVID-19 : chute conjoncturelle des émissions liée aux confinements; rebond marqué en 2021–2022.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                    # 2022 Ukraine
                                    html.Div([
                                        html.Strong("2022"),
                                        html.P("Guerre en Ukraine : tensions gazières, retour transitoire du charbon en Europe et en Asie; accélération parallèle des renouvelables.", className="mb-0")
                                    ], className="p-2 border rounded", style={"minWidth": "260px", "backgroundColor": "#f8f9fa"}),
                                ], style={"display": "flex", "gap": "12px", "overflowX": "auto"}),
                            ], className="p-2"
                        ), width=12)
                ], className="mb-4"),

                #  Scénarios et politiques 
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Scénario inertiel (business-as-usual)", className="text-danger"),
                            html.P(
                                "Prolongation des tendances actuelles : demande énergétique en hausse, substitution lente des fossiles;"
                                " trajectoires d'émissions globalement croissantes à moyen terme.")
                        ])
                    ], color="light", style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "height": "100%"}), width=6),
                    dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H6("Scénario de transition accélérée", className="text-success"),
                            html.P(
                                "Accélération des politiques net zero, électrification des usages finaux, capture et stockage du carbone,"
                                " montée du nucléaire et des renouvelables : trajectoire compatible avec une inflexion des émissions.")
                        ])
                    ], color="light", style={"border": "1px solid #dee2e6", "borderRadius": "0.5rem", "height": "100%"}), width=6),
                ], className="mb-4"),

                #  Conclusion 
                dbc.Row([
                    dbc.Col(
                        html.Div(
                            "Les tendances passées et les événements géopolitiques récents expliquent la concentration des émissions et la poursuite de la hausse à court/moyen terme."
                            " Les trajectoires futures dépendront de l'intensité des politiques climatiques, des choix industriels (relocalisations, mix énergétique)"
                            " et des innovations (efficacité, stockage, hydrogène).",
                            className="text-muted fst-italic text-center p-3 border rounded",
                            style={
                                "backgroundColor": "#f8f9fa",
                                "border": "1px solid #dee2e6",
                                "color": "#212529",
                                "maxWidth": "90%",
                                "margin": "auto",
                                "boxShadow": "0 0 10px rgba(0,0,0,0.05)",
                                "fontSize": "0.95rem",
                                "lineHeight": "1.5"
                            }
                        ), width=12)
                ], className="mb-4"),
            ])
        ]),
        # (fin des onglets)
    ])
], fluid=True)


#  Helper functions for dynamic indicator computation 
def compute_emissions_per_capita(annee):
    if "Population" in df_pays.columns:
        return df_pays[annee] / df_pays["Population"]
    else:
        # Placeholder random scaling if population data not available
        return df_pays[annee] / (df_pays[annee].max() / 10)

def compute_variation_since_1990(annee):
    if "1990" in df_pays.columns and annee in df_pays.columns:
        return ((df_pays[annee] - df_pays["1990"]) / df_pays["1990"]) * 100
    else:
        return pd.Series([None] * len(df_pays))

# Callback interactivité : mise à jour de la carte, du top 10 et de l'insight par année + indicateur sélectionnés
@app.callback(
    Output("carte-emissions", "figure"),
    Output("graph-top10", "figure"),
    Output("insight-pays", "children"),
    Input("dropdown-annee", "value"),
    Input("dropdown-indicateur", "value")
)
def update_visuals(annee, indicateur):
    df_temp = df_pays.copy()
    if indicateur == "per_capita":
        df_temp[annee] = compute_emissions_per_capita(annee)
        title_suffix = "(par habitant)"
    elif indicateur == "variation":
        df_temp[annee] = compute_variation_since_1990(annee)
        title_suffix = "(variation depuis 1990, %)"
    else:
        title_suffix = "(millions de tonnes)"

    def build_dynamic_carte():
        fig = px.choropleth(
            df_temp,
            locations="Country Code",
            color=annee,
            hover_name="Country Name",
            color_continuous_scale="Viridis",
            title=f"Émissions de CO₂ par pays en {annee} {title_suffix}",
            labels={annee: title_suffix}
        )
        fig.update_layout(template="plotly_white", height=800, margin=dict(l=0, r=0, t=60, b=0))
        fig.update_geos(
            showcountries=True,
            countrycolor="lightgray",
            showcoastlines=True,
            coastlinecolor="white",
            showland=True,
            landcolor="#f9f9f9",
            projection_type="natural earth",
            bgcolor="rgba(240,240,240,0)"
        )
        return fig

    def build_dynamic_top10():
        top10 = df_temp[["Country Name", annee]].dropna().sort_values(annee, ascending=False).head(10)
        fig = px.bar(
            top10.sort_values(annee, ascending=True),
            y="Country Name",
            x=annee,
            orientation="h",
            title=f"Top 10 des pays {title_suffix} en {annee}",
            text=top10[annee].round(2).astype(str),
            color=annee,
            color_continuous_scale="Blues"
        )
        fig.update_layout(template="plotly_white", height=650, margin=dict(l=220, r=40, t=70, b=50))
        return fig

    fig_carte = build_dynamic_carte()
    fig_top10 = build_dynamic_top10()
    insight_text = generate_insights_pays(str(annee))
    return fig_carte, fig_top10, insight_text

# Export CSV callback
import io
import csv
from dash import no_update
from dash.exceptions import PreventUpdate

@app.callback(
    Output("download-kpi-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    prevent_initial_call=True
)
def export_kpi_to_csv(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Indicateur", "Valeur"])
    writer.writerow(["Émissions totales mondiales en 2023 (MtCO₂)", total_mondial_2023_fmt])
    writer.writerow(["Variation des émissions depuis 1990 (%)", f"{variation_pct:.2f}"])
    writer.writerow(["CAGR (1990–2023) (%)", f"{cagr:.2f}"])
    writer.writerow(["Année du pic d’émission", annee_pic])
    writer.writerow(["% de pays ayant réduit leurs émissions depuis 1990", f"{pct_reduction:.2f}"])
    writer.writerow(["Ratio 80/20 (part des pays cumulant 80 % des émissions)", f"{ratio_80_20:.2f}"])
    writer.writerow([])
    writer.writerow(["Résumé global", generate_insights_global()])

    buffer.seek(0)
    return dict(content=buffer.getvalue(), filename="indicateurs_CO2.csv")

#  Callback Top 10 projection dynamique et insight automatique 
@app.callback(
    Output("graph-top-projection", "figure"),
    Output("insight-projection", "children"),
    Input("dropdown-horizon", "value")
)
def update_top_projection(horizon):
    col_proj = f"Projection_{horizon}"
    df_temp = df_cagr_pays[["Country Name", col_proj]].dropna().sort_values(col_proj, ascending=False)
    top10 = df_temp.head(10)
    total_proj = df_temp[col_proj].sum()
    part_top10 = top10[col_proj].sum() / total_proj * 100

    fig = px.bar(
        top10.sort_values(col_proj, ascending=True),
        y="Country Name",
        x=col_proj,
        orientation="h",
        title=f"Top 10 des pays avec les émissions projetées les plus élevées en {horizon} (millions de tonnes)",
        text=top10[col_proj].round(0).astype(int).astype(str) + " M",
        color=col_proj,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=220, r=40, t=70, b=50),
        xaxis_title="Émissions projetées (millions de tonnes)",
        yaxis_title="Pays",
        title_font=dict(size=13),
        xaxis_title_font=dict(size=12),
        yaxis_title_font=dict(size=12),
        xaxis_tickfont=dict(size=11),
        yaxis_tickfont=dict(size=11),
    )
    fig.update_traces(textposition="outside", textfont=dict(size=10))

    # Insight automatique
    if len(top10) >= 3:
        p1, p2, p3 = top10.iloc[0]["Country Name"], top10.iloc[1]["Country Name"], top10.iloc[2]["Country Name"]
        v1 = top10.iloc[0][col_proj] / total_proj * 100
        v2 = top10.iloc[1][col_proj] / total_proj * 100
        v3 = top10.iloc[2][col_proj] / total_proj * 100
        insight = (
            f"En {horizon}, {p1} resterait le principal émetteur mondial de CO₂ ({v1:.1f} % des émissions projetées), "
            f"suivi de {p2} ({v2:.1f} %) et de {p3} ({v3:.1f} %). "
            f"Les 10 premiers pays cumuleraient environ {part_top10:.1f} % des émissions globales estimées."
        )
    else:
        insight = f"Données insuffisantes pour établir le Top 10 en {horizon}."
    return fig, insight

#  6) LANCEMENT 
import webbrowser
import os

if __name__ == "__main__":
    # Ouvre le navigateur uniquement lors du processus principal en mode debug
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=True)
