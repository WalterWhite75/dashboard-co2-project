#  Dashboard CO₂ — Analyse mondiale des émissions (1970–2023)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Plotly Dash](https://img.shields.io/badge/Dash-Plotly-brightgreen?logo=plotly)](https://plotly.com/dash/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Made with VS Code](https://img.shields.io/badge/Made%20with-VSCode-blue?logo=visualstudiocode)](https://code.visualstudio.com/)

---

##  Objectif du projet

Ce projet vise à **analyser et visualiser les émissions mondiales de CO₂** entre 1970 et 2023 à l’aide d’un **tableau de bord interactif** réalisé avec **Dash, Plotly et Pandas**.  
Il inclut également des **projections futures (2030–2050)** basées sur le **CAGR** (taux de croissance annuel composé) ainsi qu’une **analyse géopolitique** des tendances.

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
