# Indian Regional Music Industry Analytics & Popularity Predictor

Data Analytics + AI project analyzing 7,000 Spotify tracks across 7 Indian
film industries (Bollywood, Tollywood, Kollywood, Sandalwood, Mollywood,
Punjabi, English) to uncover popularity trends and predict genre &
popularity using machine learning.

**Dataset:** [Spotify India Music — Kaggle](https://www.kaggle.com/datasets/bhanuprakashchegondi/spotify-india-music)

---

## Problem Statement

Indian regional film music is rarely analyzed industry-by-industry. This
project performs exploratory data analysis across all 7 industries,
engineers meaningful features (temporal, collaboration, text-based),
trains ML models to classify genre and predict popularity, and builds an
artist collaboration network to reveal how artists work within and across
industries.

---

## Dataset Description

| Field        | Description                                              |
|--------------|-----------------------------------------------------------|
| Track ID     | Unique Spotify track identifier                           |
| Track Name   | Song title                                                 |
| Artist(s)    | List of contributing artists                               |
| Album        | Album/soundtrack name                                      |
| Release Date | YYYY-MM-DD                                                  |
| Popularity   | Spotify popularity score, 0–100                            |
| Genre        | Film industry (bollywood, tollywood, kollywood, sandalwood, mollywood, punjabi, english) |

- 7,000 rows × 8 columns, no missing values
- Year span: 1958–2025
- Popularity range: 10–78 (mean ≈ 42)

---

## Technologies Used

- **Python** (pandas, numpy) — data cleaning & feature engineering
- **Matplotlib, Seaborn** — EDA visualizations
- **Scikit-learn** — Random Forest classification, Linear/Random Forest regression, TF-IDF
- **XGBoost** — gradient-boosted regression
- **NetworkX** — artist collaboration network graphs
- **Flask / Streamlit** — interactive dashboard (optional, bonus)

---

## Setup & Run Instructions

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the full ML pipeline**
```bash
python Spoorthi_H_Rao_Indian_Music_Analytics.py
```
This loads and cleans the data, engineers features, runs EDA (saves 15
charts to `figures/`), trains and evaluates all models, builds the
collaboration network, and exports `processed_dataset.csv`,
`model_results.json`, and trained model files (`.pkl`).

Runtime: ~1–2 minutes on a standard laptop.

**3. (Optional) Launch the interactive dashboard**
```bash
python backend/server.py
```
Open `http://localhost:8000` to explore the EDA charts, ML results,
popularity predictor, and collaboration network interactively.

---

## Approach Summary

**Feature Engineering:** release year/month/decade, number of collaborating
artists, collaboration flag, track name length/word count, "feat" flag,
TF-IDF (unigrams + bigrams) on track names.

**ML Task 1 — Genre Classification:** Random Forest Classifier using
numeric + TF-IDF features. Test accuracy ≈ 47% (7-class problem, random
baseline ≈ 14%).

**ML Task 2 — Popularity Regression:** Compared Linear Regression, Random
Forest, and XGBoost.

| Model             | RMSE  | R²     |
|-------------------|-------|--------|
| Linear Regression | 11.53 | 0.1916 |
| Random Forest     | 7.13  | 0.6913 |
| **XGBoost**       | **7.04** | **0.6987** |

**Artist Collaboration Network:** built with NetworkX — nodes are artists,
edges represent shared track appearances, visualized per-industry and
across industries.

---

## Key Insights

1. Bollywood has the highest average popularity (~60), notably above
   Tollywood/Kollywood (~41–42).
2. Release volume has surged since 2016, reflecting streaming-era growth.
3. Collaborative tracks are common — average of ~2 artists per track.
4. XGBoost significantly outperforms Linear Regression (R² 0.70 vs 0.19),
   showing popularity depends on non-linear feature interactions.
5. Cross-industry artist collaboration is relatively rare — most artists
   stay within their regional industry.

---

## Project Structure

```
├── Spoorthi_H_Rao_Indian_Music_Analytics.py   # Full ML pipeline (run this first)
├── requirements.txt
├── README.md
├── Spoorthi_H_Rao_PROJECT_REPORT.docx
├── spotify_dataset.csv
├── processed_dataset.csv      # generated
├── model_results.json         # generated
├── figures/                   # 15 generated charts
├── backend/server.py          # optional dashboard API
└── frontend/index.html        # optional dashboard UI
```

---

## Author

Spoorthi H Rao
Data Analytics & AI Internship
