# 🎵 Indian Regional Music Industry Analytics & Popularity Predictor

> **IBM Watson Studio / AutoAI Internship Project**  
> End-to-end Data Analytics + Machine Learning pipeline on 7,000 Spotify tracks  
> spanning 7 Indian film industries.

---

## 📌 Problem Statement

Indian regional film music is one of the most diverse and data-rich segments of  
the global music industry, yet analytics tooling rarely treats each regional  
industry (Bollywood, Tollywood, Kollywood, Sandalwood, Mollywood, Punjabi, English)  
as distinct entities. This project addresses that gap by:

1. Performing systematic **Exploratory Data Analysis** on cross-industry  
   popularity and release patterns.
2. Engineering meaningful **features** (temporal, collaboration, text).
3. Training **classification** and **regression** models to predict genre and  
   popularity score respectively.
4. Building an **artist collaboration network** to reveal intra- and  
   cross-industry co-working patterns.
5. Packaging everything into a full-stack **Flask REST API & Web Dashboard** (with an optional legacy **Streamlit app**) for interactive exploration.

---

## 📂 Folder / File Structure

```
ibm internship/
│
├── spotify_dataset.csv          # Raw dataset (7,000 tracks)
├── processed_dataset.csv        # Feature-engineered dataset (generated)
├── model_results.json           # Serialised model metrics (generated)
│
├── Spoorthi_H_Rao_Indian_Music_Analytics.py # Full ML pipeline script
├── main.py                      # Pipeline wrapper (runs Spoorthi script)
├── app.py                       # Legacy Streamlit dashboard (optional)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── PROJECT_REPORT.docx          # Formal internship report
│
├── backend/
│   └── server.py                # Flask REST API — run this to start the app
│
├── frontend/
│   └── index.html               # Full SPA dashboard (HTML + CSS + JS)
│
└── figures/                     # All generated charts (15 PNGs)
    ├── 01_popularity_distribution.png
    ├── 02_avg_popularity_per_genre.png
    ├── 03_popularity_trend_by_decade.png
    ├── 04_release_trend_by_year.png
    ├── 05_release_trend_genre_area.png
    ├── 06_top_artists_per_genre.png
    ├── 07_collaboration_distribution.png
    ├── 08_correlation_heatmap.png
    ├── 09_confusion_matrix.png
    ├── 10_clf_feature_importance.png
    ├── 11_regression_model_comparison.png
    ├── 12_predicted_vs_actual.png
    ├── 13_xgb_feature_importance.png
    ├── 14_collaboration_network_per_genre.png
    └── 15_cross_industry_network.png
```

---

## 📊 Dataset Description

🔗 **Dataset Source:** [Spotify India Music Dataset on Kaggle](https://www.kaggle.com/datasets/bhanuprakashchegondi/spotify-india-music)

| Field        | Description                                                   |
|--------------|---------------------------------------------------------------|
| Track ID     | Unique Spotify track identifier                               |
| Track Name   | Song title                                                    |
| Artist(s)    | Python-list string of all contributing artists                |
| Album        | Album or soundtrack name                                      |
| Release Date | ISO 8601 date (YYYY-MM-DD)                                    |
| Cover Image  | Spotify CDN URL for album art                                 |
| Popularity   | Spotify popularity score 0–100 (higher = more popular)        |
| Genre        | Film industry: bollywood / tollywood / kollywood / sandalwood / mollywood / punjabi / english |

- **Raw Size:** 7,000 rows × 8 columns
- **Cleaned Size:** 6,697 rows (after cleaning and unparseable date filtering)
- **Year Span:** 1958 – 2025  
- **Popularity Range:** 10 – 78 (mean ≈ 42)

---

## 🚀 Setup & Run Instructions

### 1. Prerequisites

- Python 3.9 or higher (tested on 3.12)
- pip (bundled with Python)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ML pipeline (generates all figures + processed data)

```bash
python main.py
```

This will:
- Load and clean the raw CSV
- Engineer all features
- Produce 15 publication-quality charts in `figures/`
- Train and evaluate all models (classification + regression)
- Build the artist collaboration network
- Export `processed_dataset.csv` and `model_results.json`

Expected runtime: **~60 – 120 seconds** on a standard laptop.

### 4. Launch the full-stack web app ⭐ Recommended

```bash
python backend/server.py
```

Then open **http://localhost:8000** in your browser.

The Flask server:
- Serves the polished HTML/JS/CSS frontend from `frontend/index.html`
- Exposes a full REST API at `http://localhost:8000/api/...`
- Powers 8 interactive pages: Overview, EDA, Trends Explorer, Top Artists,
  ML Results, Popularity Predictor, Collaboration Network, Track Browser

### 5. (Optional) Legacy Streamlit dashboard

```bash
python -m streamlit run app.py
```

Open http://localhost:8501. Use this if you prefer the Streamlit interface.

> **Note — Windows PATH:** If `streamlit` is not recognised, use `python -m streamlit run app.py`.

---

## 🔬 Summary of Approach

### Exploratory Data Analysis
- Genre-wise popularity distribution (histograms + bar charts)
- Average popularity per industry
- Popularity trends across decades (line chart)
- Release volume by year and genre (area chart)
- Top 10 most prolific artists per industry
- Collaboration count distribution
- Feature correlation heatmap

### Feature Engineering
| Feature           | How derived                                          |
|-------------------|------------------------------------------------------|
| `release_year`    | Extracted from Release Date                          |
| `release_month`   | Extracted from Release Date                          |
| `release_decade`  | `release_year // 10 * 10`                            |
| `n_artists`       | `len(parse_artists(Artist(s)))`                      |
| `is_collab`       | 1 if `n_artists > 1`, else 0                         |
| `track_name_len`  | `len(Track Name)`                                    |
| `track_word_count`| `len(Track Name.split())`                            |
| `has_feat`        | 1 if "feat" / "ft" appears in track name             |
| TF-IDF (500 dims) | Unigram + bigram TF-IDF on Track Name (classification only) |

### ML Task 1 — Genre Classification
- **Model:** Random Forest Classifier (200 estimators, max_depth 20)
- **Features:** Numeric (8) + TF-IDF (500)
- **Test Accuracy:** ~46.8%  
  *(Genre is genuinely hard to predict from non-audio features alone;  
  the model still learns clear signals for Bollywood and English)*

### ML Task 2 — Popularity Regression
| Model             | RMSE   | R²     |
|-------------------|--------|--------|
| Linear Regression | 11.53  | 0.1916 |
| Random Forest     | 7.13   | 0.6913 |
| **XGBoost**       | **7.04** | **0.6987** |

XGBoost is the best model; `genre_encoded` and `release_year` are its most  
important features, confirming that recency and industry are the primary  
drivers of Spotify popularity.

### Artist Collaboration Network
Built with **NetworkX**:
- Nodes = artists; edges = shared track appearances
- Edge weight = number of co-appearances
- Per-industry networks (min 2 co-appearances)
- Cross-industry network showing hub artists who bridge industries

---

## 🔑 Key Results & Insights

1. **Bollywood dominates popularity** — mean score 59.7, ~15 points above  
   Tollywood and Kollywood (≈41.5 each).
2. **English-language tracks score lowest** (~32.3) despite global reach,  
   possibly because the dataset captures South-Asian Spotify listeners.
3. **Release volume has surged post-2016** — over 600 tracks in 2022 and  
   2023 alone, reflecting streaming-era growth across all industries.
4. **Collaborative tracks are common** — mean of 2.05 artists per track;  
   some tracks credit up to 10 artists.
5. **Non-linear models dominate** — XGBoost (R² ≈ 0.70) far outperforms  
   Linear Regression (R² ≈ 0.19), confirming that popularity prediction  
   requires capturing interaction effects.
6. **Cross-industry collaborations are sparse** — most artists stay within  
   their regional industry, though Bollywood artists occasionally feature  
   on Punjabi tracks.

---

## 🏢 IBM Watson Studio / AutoAI Mapping

This project is designed to map cleanly onto the IBM Watson Studio workflow:

| Step                    | Watson Studio Equivalent                    |
|-------------------------|---------------------------------------------|
| Data loading & cleaning | **Data Refinery** — shape, filter, transform |
| EDA charts              | **Cognos Dashboard Embedded** visualisations |
| Feature engineering     | **Data Refinery flows** + Python notebook   |
| Model training          | **AutoAI** experiment (auto-selects best algorithm) |
| Model evaluation        | AutoAI pipeline leaderboard (RMSE / R²)     |
| Deployment              | **Watson Machine Learning** online endpoint  |
| Dashboard               | Streamlit app or **Cognos Analytics** report|

---

## 👤 Author

Internship Project — Data Analytics & AI Track  
IBM Watson Studio Platform
