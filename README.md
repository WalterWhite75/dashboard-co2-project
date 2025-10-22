#  Dashboard CO₂ — Analyse mondiale des émissions (1970–2023)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Plotly Dash](https://img.shields.io/badge/Dash-Plotly-brightgreen?logo=plotly)](https://plotly.com/dash/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Made with VS Code](https://img.shields.io/badge/Made%20with-VSCode-blue?logo=visualstudiocode)](https://code.visualstudio.com/)

---

##  Objectif du projet

Ce projet a pour but d’**analyser et de visualiser l’évolution des émissions mondiales de CO₂** entre 1970 et 2023 à travers un **tableau de bord interactif professionnel** développé en **Python (Dash, Plotly, Pandas)**.  
Il inclut également des **projections futures (2030–2050)** basées sur le **taux de croissance annuel composé (CAGR)** et une **analyse géopolitique** contextualisant les dynamiques d’émissions.

---

##  Contexte

Réalisé dans le cadre du **Master 2 Big Data, Analyse et Business Intelligence** à l’Université **Sorbonne Paris Nord**, ce projet illustre un **pipeline complet de data science** :
1. **ETL (Extraction, Transformation, Chargement)** des données CO₂ issues de la Banque mondiale  
2. **Nettoyage et contrôle qualité** des données  
3. **Analyse et visualisation** via un dashboard multi-onglets  
4. **Storytelling analytique** et **export automatisé des KPI**

---

##  Structure du projet

```bash
dashboard_co2_project/
│
├── data/
│   ├── DonnéesC02.csv              # Données brutes (Banque mondiale)
│   └── DonnéesC02_clean.csv        # Données nettoyées
│
├── scripts/
│   ├── CO2.py                      # Script de nettoyage et préparation (ETL)
│   └── dashboard_co2.py            # Tableau de bord interactif (Dash)
│
├── requirements.txt                # Dépendances Python
└── README.md                       # Documentation du projet
