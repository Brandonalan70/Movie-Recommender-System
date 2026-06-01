# Movie Recommender System

A multi-source movie recommendation system that integrates data from IMDb, MovieLens, and The Movie Database (TMDb) to deliver personalized recommendations using two complementary approaches:

- **Collaborative Filtering (CF)** — user-based filtering for existing users, leveraging the preferences of similar users
- **Content-Based Filtering (CBF)** — feature-based filtering for new users, using genre and metadata similarity

## Features

- Multi-source data integration with a mediated schema for cross-source normalization
- User-based collaborative filtering (cosine similarity, configurable neighbor count)
- Content-based filtering using genre multi-hot encoding and numeric metadata features
- Comprehensive evaluation suite: Precision, Recall, F1, NDCG, MAP, Hit Rate, MRR, Coverage
- Interactive CLI demo supporting both new and existing users

## Project Structure

```
Movie-Recommender-System/
├── movie_recommender/      # Core library (importable package)
│   ├── __init__.py
│   ├── config.py           # Path and environment configuration
│   ├── data_load.py        # Raw data loaders for each source
│   ├── mediated.py         # Source-to-mediated schema transformation
│   ├── integration.py      # Cross-source movie deduplication and merging
│   ├── features.py         # Feature matrix construction (content + CF)
│   ├── recommender.py      # MovieRecommender class (CF + CBF)
│   └── utils.py            # Shared utility functions
├── scripts/                # Runnable entry points
│   ├── prep_data.py        # Data preparation pipeline (run once)
│   ├── demo.py             # Interactive CLI demo
│   ├── evaluation.py       # Full model evaluation suite
│   └── comparison.py       # CF vs CBF comparison visualizations
├── data/
│   ├── raw/                # Input CSVs: IMDb, MovieLens, TMDb (not tracked in git)
│   └── processed/          # Pipeline-generated intermediates (not tracked in git)
├── results/                # Evaluation charts and visualizations
├── report/
│   └── Movie Recommender System Report.docx
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone <repo-url>
cd Movie-Recommender-System
pip install -e .
```

The `-e .` install reads `pyproject.toml` and makes `movie_recommender` importable from anywhere in the project, so all scripts in `scripts/` work without any path configuration.

## Data Setup

Place the following CSV files in `data/raw/` before running the pipeline:

| File | Source |
|------|--------|
| `basics_imdb.csv` | [IMDb Datasets](https://datasets.imdbws.com/) — `title.basics.tsv` converted to CSV |
| `ratings_imdb.csv` | [IMDb Datasets](https://datasets.imdbws.com/) — `title.ratings.tsv` converted to CSV |
| `movies_ml.csv` | [MovieLens](https://grouplens.org/datasets/movielens/) — `movies.csv` |
| `ratings_ml.csv` | [MovieLens](https://grouplens.org/datasets/movielens/) — `ratings.csv` |
| `movies_tmdb.csv` | [TMDb](https://www.themoviedb.org/documentation/api) — exported movie metadata |

For a TMDb API key, create a `.env` file at the project root:

```
TMDB_API_KEY=your_key_here
```

## Usage

### 1. Prepare the data

Run the pipeline once to build all intermediate CSVs in `data/processed/`:

```bash
python scripts/prep_data.py
```

### 2. Run the interactive demo

```bash
python scripts/demo.py
```

The demo prompts you to choose between two modes:

- **Existing user** — enter a MovieLens `userId` to get CF-based recommendations
- **New user** — enter 3–5 favorite movie titles to get CBF-based recommendations

```
==== Movie Recommender Demo ====
1. Recommend for existing MovieLens userId
2. Recommend for new user (enter favorite titles)
3. Quit
Choice: 2

Enter 3–5 favorite movie titles, comma-separated:
> The Matrix, Inception, Interstellar

[Recommendations for new user (content-based)]
1. Arrival (2016) | Genres: drama, science fiction, thriller | movie_id=4821
2. Blade Runner 2049 (2017) | Genres: drama, science fiction | movie_id=7103
...
```

### 3. Evaluate the models

```bash
python scripts/evaluation.py   # full evaluation suite — generates charts in results/
python scripts/comparison.py   # CF vs CBF comparison charts
```

## How It Works

### Data Pipeline

```
Raw CSVs (IMDb + MovieLens + TMDb)
        │
        ▼
   data_load.py         Load each source into pandas DataFrames
        │
        ▼
   mediated.py          Normalize to unified schema
                        (title_norm, year, genres_norm, rating_value, ...)
        │
        ▼
   integration.py       Deduplicate movies by (title_norm, year)
                        Compute weighted ratings across sources
                        Map MovieLens ratings to integrated movie IDs
        │
        ▼
   data/processed/      integrated_movies.csv
                        movielens_ratings_integrated.csv
```

### Recommendation Strategies

| Scenario | Method | Details |
|----------|--------|---------|
| Existing user | User-based CF | Find top-50 similar users by cosine similarity on the rating matrix; aggregate weighted ratings for unseen movies |
| New user | Content-based | Average feature vectors of favorite movies; rank all movies by cosine similarity to that profile |

Content features include genre multi-hot encoding plus scaled numeric attributes (release year, average rating, popularity).

## Evaluation Results (k=10)

| Metric | CF Model | CBF Model |
|--------|----------|-----------|
| Precision | 0.180 | 0.110 |
| Recall | 0.120 | 0.140 |
| F1 | 0.140 | 0.120 |
| NDCG | 0.320 | 0.230 |
| MAP | 0.170 | 0.110 |
| Hit Rate | 0.550 | 0.430 |
| MRR | 0.370 | 0.260 |
| Coverage | 0.280 | 0.680 |

CF dominates on accuracy-oriented metrics; CBF excels at catalog coverage and handles new users with no rating history. A hybrid approach combining both is the recommended production strategy.

Evaluation charts are saved in [`results/`](results/).

## Report

See [`report/Movie Recommender System Report.docx`](report/Movie%20Recommender%20System%20Report.docx) for the full academic write-up covering methodology, experimental design, and findings.

## License

MIT
