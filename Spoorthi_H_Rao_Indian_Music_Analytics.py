"""
============================================================
  Indian Regional Music Industry Analytics & Popularity Predictor
  Full ML Pipeline — EDA, Feature Engineering, Classification,
  Regression, Artist Collaboration Network
============================================================
  Dataset : spotify_dataset.csv  (7,000 tracks, 7 industries)
  Author  : Internship Project
  Python  : 3.9+
============================================================
"""

# ── Standard library ───────────────────────────────────────
import ast
import os
import warnings
import itertools
from collections import Counter

# ── Data / numerics ────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── ML — preprocessing & feature engineering ───────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ── ML — classification ────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

# ── ML — regression ────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ── Network analysis ───────────────────────────────────────
import networkx as nx

warnings.filterwarnings("ignore")

# ── Global style ───────────────────────────────────────────
PALETTE = {
    "bollywood" : "#E63946",
    "tollywood" : "#F4A261",
    "kollywood" : "#2A9D8F",
    "sandalwood": "#457B9D",
    "mollywood" : "#8338EC",
    "punjabi"   : "#FB8500",
    "english"   : "#3A86FF",
}
GENRE_ORDER = list(PALETTE.keys())
COLOR_LIST  = list(PALETTE.values())

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"      : 120,
    "axes.titleweight": "bold",
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
})

# Output directory for saved figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def savefig(name: str):
    """Save current figure to the figures/ directory."""
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 1 — DATA LOADING & CLEANING")
print("="*60)

df = pd.read_csv("spotify_dataset.csv")
print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

# Parse Release Date → datetime
df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")

# Drop rows where Release Date is completely unparseable (none expected)
df.dropna(subset=["Release Date"], inplace=True)

# Normalise genre to lowercase (already lowercase, but be safe)
df["Genre"] = df["Genre"].str.strip().str.lower()

# Popularity — ensure numeric
df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
df.dropna(subset=["Popularity"], inplace=True)

print(f"Clean dataset: {len(df):,} rows")
print(f"Genres       : {sorted(df['Genre'].unique())}")
print(f"Date range   : {df['Release Date'].dt.year.min()} – {df['Release Date'].dt.year.max()}")
print(f"Popularity   : {df['Popularity'].min():.0f} – {df['Popularity'].max():.0f}  "
      f"(mean {df['Popularity'].mean():.1f})")


# ══════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 2 — FEATURE ENGINEERING")
print("="*60)

# 2.1 Temporal features
df["release_year"]   = df["Release Date"].dt.year.astype(int)
df["release_month"]  = df["Release Date"].dt.month.astype(int)
df["release_decade"] = (df["release_year"] // 10 * 10).astype(int)

# 2.2 Artist-list parsing — count collaborators per track
def parse_artists(raw: str) -> list:
    """Safely parse the Python-list-like Artist(s) string."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return [a.strip() for a in str(raw).split(",")]

df["artist_list"]  = df["Artist(s)"].apply(parse_artists)
df["n_artists"]    = df["artist_list"].apply(len)
df["is_collab"]    = (df["n_artists"] > 1).astype(int)

# 2.3 Text features from Track Name
df["track_name_len"]    = df["Track Name"].str.len()
df["track_word_count"]  = df["Track Name"].str.split().str.len()
df["has_feat"]          = df["Track Name"].str.lower().str.contains(
                              r"\b(feat|ft)\b", regex=True).astype(int)

# 2.4 Genre label encoding (for ML models that need numeric target)
le = LabelEncoder()
df["genre_encoded"] = le.fit_transform(df["Genre"])

print("New columns added:")
for c in ["release_year","release_decade","n_artists","is_collab",
          "track_name_len","track_word_count","has_feat","genre_encoded"]:
    print(f"  {c:25s}  sample: {df[c].iloc[0]}")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 3 — EXPLORATORY DATA ANALYSIS")
print("="*60)

# ── 3.1 Popularity distribution by genre ──────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for genre in GENRE_ORDER:
    data = df.loc[df["Genre"] == genre, "Popularity"]
    ax.hist(data, bins=25, alpha=0.55, label=genre.capitalize(),
            color=PALETTE[genre], edgecolor="none")
ax.set_title("Popularity Score Distribution by Genre")
ax.set_xlabel("Popularity Score")
ax.set_ylabel("Track Count")
ax.legend(ncol=2)
savefig("01_popularity_distribution.png")

# ── 3.2 Average popularity per genre (bar chart) ──────────
genre_pop = df.groupby("Genre")["Popularity"].mean().reindex(GENRE_ORDER)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(GENRE_ORDER, genre_pop.values,
              color=[PALETTE[g] for g in GENRE_ORDER], edgecolor="white")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
ax.set_title("Average Popularity Score per Film Industry / Genre")
ax.set_xlabel("Industry / Genre")
ax.set_ylabel("Mean Popularity")
ax.set_ylim(0, genre_pop.max() * 1.18)
ax.set_xticklabels([g.capitalize() for g in GENRE_ORDER])
savefig("02_avg_popularity_per_genre.png")

# ── 3.3 Genre-wise popularity trend over decades ──────────
decade_pop = (df.groupby(["release_decade", "Genre"])["Popularity"]
                .mean().reset_index())
fig, ax = plt.subplots(figsize=(11, 5))
for genre in GENRE_ORDER:
    sub = decade_pop[decade_pop["Genre"] == genre]
    ax.plot(sub["release_decade"], sub["Popularity"],
            marker="o", linewidth=2, label=genre.capitalize(),
            color=PALETTE[genre])
ax.set_title("Genre-wise Popularity Trend Over Decades")
ax.set_xlabel("Release Decade")
ax.set_ylabel("Mean Popularity")
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.legend(ncol=2)
savefig("03_popularity_trend_by_decade.png")

# ── 3.4 Track release count by year ───────────────────────
year_counts = df["release_year"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(year_counts.index, year_counts.values, color="#3b82d4", width=0.8)
ax.set_title("Number of Tracks Released per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Track Count")
ax.set_xlim(year_counts.index.min() - 1, year_counts.index.max() + 1)
savefig("04_release_trend_by_year.png")

# ── 3.5 Release trend by year per genre (stacked area) ────
year_genre = (df.groupby(["release_year", "Genre"])
                .size().unstack(fill_value=0))
# Filter to years 2000+ for clarity
year_genre = year_genre[year_genre.index >= 2000]
fig, ax = plt.subplots(figsize=(12, 5))
year_genre[[g for g in GENRE_ORDER if g in year_genre.columns]].plot.area(
    ax=ax, color=[PALETTE[g] for g in GENRE_ORDER if g in year_genre.columns],
    alpha=0.75, linewidth=0)
ax.set_title("Track Releases per Genre per Year (2000 onwards)")
ax.set_xlabel("Year")
ax.set_ylabel("Track Count")
ax.legend([g.capitalize() for g in GENRE_ORDER if g in year_genre.columns],
          loc="upper left", ncol=2)
savefig("05_release_trend_genre_area.png")

# ── 3.6 Top 10 most prolific artists per genre ────────────
def top_artists(genre_name: str, n: int = 10) -> pd.Series:
    all_artists = []
    for lst in df.loc[df["Genre"] == genre_name, "artist_list"]:
        all_artists.extend(lst)
    return pd.Series(Counter(all_artists)).sort_values(ascending=False).head(n)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, genre in enumerate(GENRE_ORDER):
    top = top_artists(genre)
    axes[i].barh(top.index[::-1], top.values[::-1],
                 color=PALETTE[genre], edgecolor="none")
    axes[i].set_title(genre.capitalize())
    axes[i].set_xlabel("Track appearances")
axes[-1].set_visible(False)   # hide the 8th empty subplot
plt.suptitle("Top 10 Most Prolific Artists per Industry", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
savefig("06_top_artists_per_genre.png")

# ── 3.7 Collaboration count distribution ──────────────────
fig, ax = plt.subplots(figsize=(7, 4))
collab_counts = df["n_artists"].value_counts().sort_index()
ax.bar(collab_counts.index, collab_counts.values, color="#7c5cd8")
ax.set_title("Distribution of Number of Collaborating Artists per Track")
ax.set_xlabel("Number of Artists")
ax.set_ylabel("Track Count")
savefig("07_collaboration_distribution.png")

# ── 3.8 Correlation heatmap of numeric features ───────────
num_cols = ["Popularity", "release_year", "n_artists",
            "track_name_len", "track_word_count", "has_feat", "is_collab"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax)
ax.set_title("Feature Correlation Heatmap")
savefig("08_correlation_heatmap.png")

print("EDA plots saved.")


# ══════════════════════════════════════════════════════════════
# SECTION 4 — ML TASK 1: GENRE CLASSIFICATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 4 — ML TASK 1: GENRE CLASSIFICATION")
print("="*60)

# Feature set: numeric + TF-IDF on Track Name
NUMERIC_FEATURES = [
    "release_year", "release_month", "release_decade",
    "n_artists", "is_collab", "track_name_len",
    "track_word_count", "has_feat"
]
TEXT_FEATURE = "Track Name"
TARGET_CLF   = "genre_encoded"

X_num  = df[NUMERIC_FEATURES].values
X_text = df[TEXT_FEATURE].fillna("").values
y_clf  = df[TARGET_CLF].values

# Build a TF-IDF matrix for track names (unigrams + bigrams, top 500 features)
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                         analyzer="word", lowercase=True)
X_tfidf = tfidf.fit_transform(X_text).toarray()

# Concatenate numeric + TF-IDF
X_clf = np.hstack([X_num, X_tfidf])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# Scale numeric portion only (TF-IDF is already normalised)
scaler_clf = StandardScaler()
X_train_c[:, :len(NUMERIC_FEATURES)] = scaler_clf.fit_transform(
    X_train_c[:, :len(NUMERIC_FEATURES)])
X_test_c[:, :len(NUMERIC_FEATURES)] = scaler_clf.transform(
    X_test_c[:, :len(NUMERIC_FEATURES)])

# ── Train Random Forest classifier ────────────────────────
clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                              random_state=42, n_jobs=-1)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred_c)
print(f"\nRandom Forest Classifier — Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c,
                             target_names=[g.capitalize() for g in le.classes_]))

# ── Confusion matrix ──────────────────────────────────────
cm = confusion_matrix(y_test_c, y_pred_c)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[g.capitalize() for g in le.classes_],
            yticklabels=[g.capitalize() for g in le.classes_],
            ax=ax)
ax.set_title(f"Genre Classification — Confusion Matrix  (Acc = {acc:.3f})")
ax.set_xlabel("Predicted Genre")
ax.set_ylabel("True Genre")
plt.tight_layout()
savefig("09_confusion_matrix.png")

# ── Feature importances (top 20 numeric features) ─────────
fi = clf.feature_importances_[:len(NUMERIC_FEATURES)]
fi_df = pd.DataFrame({"Feature": NUMERIC_FEATURES, "Importance": fi})
fi_df.sort_values("Importance", ascending=True, inplace=True)

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(fi_df["Feature"], fi_df["Importance"], color="#3b82d4")
ax.set_title("Random Forest — Top Numeric Feature Importances (Classification)")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
savefig("10_clf_feature_importance.png")


# ══════════════════════════════════════════════════════════════
# SECTION 5 — ML TASK 2: POPULARITY REGRESSION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 5 — ML TASK 2: POPULARITY REGRESSION")
print("="*60)

# Feature set for regression (no TF-IDF — keeps it comparable)
REG_FEATURES = [
    "genre_encoded", "release_year", "release_month",
    "release_decade", "n_artists", "is_collab",
    "track_name_len", "track_word_count", "has_feat"
]
TARGET_REG = "Popularity"

X_reg = df[REG_FEATURES].values
y_reg = df[TARGET_REG].values

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_train_rs = scaler_reg.fit_transform(X_train_r)
X_test_rs  = scaler_reg.transform(X_test_r)

# Helper to evaluate a regressor
def evaluate_regressor(model, X_tr, X_te, y_tr, y_te,
                        name: str, scaled=True):
    Xtr = X_tr if scaled else X_train_r
    Xte = X_te if scaled else X_test_r
    model.fit(Xtr, y_tr)
    pred = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    print(f"  {name:30s}  RMSE: {rmse:.3f}   R²: {r2:.4f}")
    return pred, rmse, r2

reg_results = {}

# ── 5.1 Linear Regression ─────────────────────────────────
pred_lr, rmse_lr, r2_lr = evaluate_regressor(
    LinearRegression(), X_train_rs, X_test_rs, y_train_r, y_test_r,
    "Linear Regression")
reg_results["Linear Regression"] = {"RMSE": rmse_lr, "R²": r2_lr}

# ── 5.2 Random Forest Regressor ───────────────────────────
pred_rf, rmse_rf, r2_rf = evaluate_regressor(
    RandomForestRegressor(n_estimators=200, max_depth=15,
                          random_state=42, n_jobs=-1),
    X_train_rs, X_test_rs, y_train_r, y_test_r,
    "Random Forest")
reg_results["Random Forest"] = {"RMSE": rmse_rf, "R²": r2_rf}

# ── 5.3 XGBoost Regressor ─────────────────────────────────
xgb_model = XGBRegressor(n_estimators=300, max_depth=6,
                          learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=42,
                          verbosity=0)
pred_xgb, rmse_xgb, r2_xgb = evaluate_regressor(
    xgb_model, X_train_rs, X_test_rs, y_train_r, y_test_r,
    "XGBoost")
reg_results["XGBoost"] = {"RMSE": rmse_xgb, "R²": r2_xgb}

# ── Model comparison bar chart ─────────────────────────────
res_df = pd.DataFrame(reg_results).T.reset_index()
res_df.columns = ["Model", "RMSE", "R²"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ["#3b82d4", "#2a9d8f", "#e63946"]
axes[0].bar(res_df["Model"], res_df["RMSE"], color=colors)
axes[0].set_title("Model Comparison — RMSE (lower is better)")
axes[0].set_ylabel("RMSE")
axes[0].set_ylim(0, res_df["RMSE"].max() * 1.25)
for i, v in enumerate(res_df["RMSE"]):
    axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

axes[1].bar(res_df["Model"], res_df["R²"], color=colors)
axes[1].set_title("Model Comparison — R² (higher is better)")
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, min(res_df["R²"].max() * 1.25, 1.0))
for i, v in enumerate(res_df["R²"]):
    axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
savefig("11_regression_model_comparison.png")

# ── Predicted vs Actual — best model (XGBoost) ────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test_r, pred_xgb, alpha=0.3, color="#e63946",
           edgecolors="none", s=15, label="XGBoost predictions")
lims = [min(y_test_r.min(), pred_xgb.min()) - 1,
        max(y_test_r.max(), pred_xgb.max()) + 1]
ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
ax.set_title("Predicted vs Actual Popularity (XGBoost)")
ax.set_xlabel("Actual Popularity")
ax.set_ylabel("Predicted Popularity")
ax.legend()
savefig("12_predicted_vs_actual.png")

# ── Feature importance for XGBoost ────────────────────────
xgb_fi = pd.Series(xgb_model.feature_importances_,
                    index=REG_FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(xgb_fi.index, xgb_fi.values, color="#e63946")
ax.set_title("XGBoost — Feature Importances (Regression)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
savefig("13_xgb_feature_importance.png")

print("\nRegression results summary:")
print(res_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════
# SECTION 6 — ARTIST COLLABORATION NETWORK
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 6 — ARTIST COLLABORATION NETWORK")
print("="*60)

# Build co-artist edges for each genre
def build_genre_graph(genre_name: str, min_edge_weight: int = 2) -> nx.Graph:
    """
    Build a weighted undirected graph where nodes are artists and
    an edge between two artists means they appeared on the same track.
    min_edge_weight: only include edges with at least this many co-appearances.
    """
    G = nx.Graph()
    sub = df[df["Genre"] == genre_name]
    for artists in sub["artist_list"]:
        for a1, a2 in itertools.combinations(artists, 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)
    # Prune low-weight edges
    remove = [(u, v) for u, v, d in G.edges(data=True)
              if d["weight"] < min_edge_weight]
    G.remove_edges_from(remove)
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

# ── Per-genre collaboration network visualisation ─────────
fig, axes = plt.subplots(2, 4, figsize=(22, 12))
axes = axes.flatten()
for i, genre in enumerate(GENRE_ORDER):
    G = build_genre_graph(genre, min_edge_weight=2)
    ax = axes[i]
    ax.set_facecolor("#0f0f1a")
    if len(G.nodes) == 0:
        ax.set_title(f"{genre.capitalize()} (no collabs)")
        continue
    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.8)
    # Node sizes proportional to degree
    degrees = dict(G.degree())
    node_sizes = [80 + degrees[n] * 60 for n in G.nodes()]
    # Edge widths
    weights    = [G[u][v]["weight"] for u, v in G.edges()]
    max_w      = max(weights) if weights else 1
    edge_widths = [0.5 + 2.5 * (w / max_w) for w in weights]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                            node_color=PALETTE[genre], alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                            edge_color="white", alpha=0.3, ax=ax)
    # Label only high-degree nodes (top 8)
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:8]
    labels    = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6,
                             font_color="white", ax=ax)
    ax.set_title(f"{genre.capitalize()}  "
                 f"({G.number_of_nodes()} artists, {G.number_of_edges()} edges)",
                 color="white")
    ax.set_axis_off()

axes[-1].set_visible(False)
fig.patch.set_facecolor("#0f0f1a")
plt.suptitle("Artist Collaboration Networks per Film Industry",
             fontsize=16, color="white", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("14_collaboration_network_per_genre.png")

# ── Cross-industry collaboration network ──────────────────
# Top-N artists per genre, build edges across genres
print("\nBuilding cross-industry collaboration network ...")

def top_n_artists_per_genre(genre_name: str, n: int = 20) -> set:
    all_artists = []
    for lst in df.loc[df["Genre"] == genre_name, "artist_list"]:
        all_artists.extend(lst)
    return set(pd.Series(Counter(all_artists)).nlargest(n).index)

# Assign each artist their primary genre
artist_genre = {}
for genre in GENRE_ORDER:
    for artist in top_n_artists_per_genre(genre, n=30):
        if artist not in artist_genre:
            artist_genre[artist] = genre

G_cross = nx.Graph()
for _, row in df.iterrows():
    artists = row["artist_list"]
    genre   = row["Genre"]
    for a1, a2 in itertools.combinations(artists, 2):
        if a1 in artist_genre and a2 in artist_genre:
            if G_cross.has_edge(a1, a2):
                G_cross[a1][a2]["weight"] += 1
            else:
                G_cross.add_edge(a1, a2, weight=1)

# Keep only edges with weight >= 2
remove = [(u, v) for u, v, d in G_cross.edges(data=True) if d["weight"] < 2]
G_cross.remove_edges_from(remove)
G_cross.remove_nodes_from(list(nx.isolates(G_cross)))

print(f"Cross-industry graph: {G_cross.number_of_nodes()} nodes, "
      f"{G_cross.number_of_edges()} edges")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor("#0f0f1a")
fig.patch.set_facecolor("#0f0f1a")
pos = nx.spring_layout(G_cross, seed=0, k=1.2)

node_colors = [PALETTE.get(artist_genre.get(n, "bollywood"), "#aaaaaa")
               for n in G_cross.nodes()]
degrees     = dict(G_cross.degree())
node_sizes  = [60 + degrees[n] * 80 for n in G_cross.nodes()]
weights     = [G_cross[u][v]["weight"] for u, v in G_cross.edges()]
max_w       = max(weights) if weights else 1
edge_widths = [0.3 + 2.5 * (w / max_w) for w in weights]

nx.draw_networkx_nodes(G_cross, pos, node_size=node_sizes,
                        node_color=node_colors, alpha=0.88, ax=ax)
nx.draw_networkx_edges(G_cross, pos, width=edge_widths,
                        edge_color="white", alpha=0.25, ax=ax)
top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:15]
nx.draw_networkx_labels(G_cross, pos,
                         labels={n: n for n in top_nodes},
                         font_size=7, font_color="white", ax=ax)

# Legend for industries
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE[g], label=g.capitalize())
                   for g in GENRE_ORDER]
ax.legend(handles=legend_elements, loc="lower right",
          facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax.set_title("Cross-Industry Artist Collaboration Network\n(node colour = primary industry)",
             color="white", fontsize=14, fontweight="bold")
ax.set_axis_off()
savefig("15_cross_industry_network.png")


# ══════════════════════════════════════════════════════════════
# SECTION 7 — SUMMARY TABLE & EXPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SECTION 7 — SUMMARY")
print("="*60)

print(f"\nClassification (Random Forest):")
print(f"  Accuracy : {acc:.4f}")

print(f"\nRegression Results:")
for model_name, metrics in reg_results.items():
    print(f"  {model_name:30s}  RMSE={metrics['RMSE']:.3f}  R²={metrics['R²']:.4f}")

# Export processed dataset for Streamlit app
df_export = df.drop(columns=["artist_list", "Cover Image"]).copy()
df_export["artists_clean"] = df["Artist(s)"]
df_export.to_csv("processed_dataset.csv", index=False)
print("\nProcessed dataset saved -> processed_dataset.csv")

# Export model results for Streamlit app
import json
model_results = {
    "classification_accuracy": round(float(acc), 4),
    "regression": {k: {m: round(float(v), 4) for m, v in vv.items()}
                   for k, vv in reg_results.items()}
}
with open("model_results.json", "w") as f:
    json.dump(model_results, f, indent=2)
print("Model results saved -> model_results.json")

print("\nPipeline complete. All figures saved to ./figures/")
print("="*60)

import joblib
joblib.dump(xgb_model, "popularity_model.pkl")
joblib.dump(clf, "genre_classifier.pkl")
joblib.dump(scaler_reg, "scaler_reg.pkl")
joblib.dump(scaler_clf, "scaler_clf.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Trained models saved for deployment.")
