import pandas as pd
from pathlib import Path

# 1) EXTRACT 
path = Path(__file__).resolve().parents[1] / "data" / "DonnéesC02.csv"

if not path.exists():
    raise FileNotFoundError(f"Fichier introuvable : {path}")

print(f"Lecture du fichier source : {path.name}")
df = pd.read_csv(path, skiprows=4)
print(f"Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")

# 2) TRANSFORM 

# a) Supprimer la colonne inutile "Unnamed: 69" si elle existe
if "Unnamed: 69" in df.columns:
    df = df.drop(columns=["Unnamed: 69"])
    print("Colonne 'Unnamed: 69' supprimée.")

# b) Supprimer toutes les colonnes correspondant aux années antérieures à 1970
colonnes_annees = [col for col in df.columns if col.isdigit() and int(col) < 1970]
if colonnes_annees:
    df = df.drop(columns=colonnes_annees)
    print(f"Colonnes antérieures à 1970 supprimées : {colonnes_annees}\n")

# Supprimer la colonne "2024" si elle existe
if "2024" in df.columns:
    df = df.drop(columns=["2024"])
    print("Colonne '2024' supprimée.")

# c) Supprimer les lignes entièrement vides (toutes les valeurs NaN sauf métadonnées)
df = df.dropna(how="all", subset=df.columns[4:])
print(f"Lignes vides supprimées. Lignes restantes : {df.shape[0]}")

# d) Vérifier les types
for col in df.columns[4:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# e) Supprimer les lignes sans nom de pays
df = df[df["Country Name"].notna()]

# f) Nettoyage des espaces ou incohérences dans les noms
df["Country Name"] = df["Country Name"].str.strip()

# Nettoyage supplémentaire : suppression des lignes parasites
df = df[~df["Country Name"].str.contains("Indicator", na=False)]
df = df[df["Country Name"].notna() & df["Country Code"].notna()]

# Nettoyage supplémentaire : suppression des agrégats régionaux et économiques
a_exclure = [
    "World", "IDA & IBRD total", "IBRD only", "IDA total", "Low & middle income",
    "High income", "Middle income", "Upper middle income", "Lower middle income",
    "OECD members", "Non-OECD members", "Europe & Central Asia", "Sub-Saharan Africa",
    "East Asia & Pacific", "Latin America & Caribbean", "North America", "South Asia",
    "Arab World", "European Union", "Euro area", "Small states",
    "Fragile and conflict affected situations", "Least developed countries: UN classification"
]
before_count = df.shape[0]
df = df[~df["Country Name"].isin(a_exclure)]
after_exclusion_count = df.shape[0]
print(f"Lignes supprimées pour agrégats régionaux et économiques : {before_count - after_exclusion_count}")

# Filtrage automatique : ne conserver que les lignes dont le Country Code est un code ISO valide (3 lettres majuscules)
df = df[df["Country Code"].str.match(r"^[A-Z]{3}$", na=False)]
after_code_filter_count = df.shape[0]
print(f"Nombre total de pays conservés après filtrage des codes ISO : {after_code_filter_count}")
print("Aperçu des 10 premiers noms de pays conservés :")
print(df["Country Name"].head(10).tolist())

# Réorganisation des colonnes dans l'ordre chronologique
colonnes_fixes = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
colonnes_annees = sorted([col for col in df.columns if col.isdigit()], key=int)
df = df[colonnes_fixes + colonnes_annees]

# 3) LOAD 
output_path = Path(__file__).resolve().parents[1] / "data" / "DonnéesC02_clean.csv"
df.to_csv(output_path, index=False)

print(f"\n Données nettoyées exportées : {output_path.name}")
print(f"Dimensions finales : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nAperçu du fichier final :")
print(df.head())