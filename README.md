# Climate Clustering of NOAA Weather Stations with K-Means

Groups global weather stations into climate types using daily observations
from the **NOAA Global Surface Summary of the Day (GSOD)** dataset on the
[Registry of Open Data on AWS](https://registry.opendata.aws/noaa-gsod/).

The bucket is public. No AWS account, credentials, or billing setup is
required — requests are signed as `UNSIGNED`, the programmatic equivalent of:

```bash
aws s3 ls --no-sign-request s3://noaa-gsod-pds/
```

---

## Quick start

```bash
git clone <repository-url>
cd kmeans-gsod-clustering

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # samples 600 stations for 2023 and clusters them
```

### In PyCharm

1. **File → Open** and select the project folder.
2. **Settings → Project → Python Interpreter → Add → Virtualenv**, then install
   `requirements.txt` when PyCharm offers.
3. Right-click `main.py` → **Run 'main'**.
4. Right-click the `tests/` folder → **Run 'pytest in tests'**.

### Command-line options

| Flag | Default | Purpose |
|---|---|---|
| `--year` | `2023` | Observation year (GSOD begins 1929) |
| `--stations` | `600` | Size of the random station sample |
| `--min-days` | `300` | Minimum reporting days for a station to be kept |
| `--k` | auto | Force a specific k instead of selecting one |
| `--local PATH` | – | Re-run from a saved CSV, no network needed |
| `--no-plots` | off | Skip chart generation |

---

## Project layout

```
kmeans-gsod-clustering/
├── main.py                     entry point, orchestrates the six pipeline steps
├── requirements.txt
├── pytest.ini
├── src/
│   ├── config.py               verified GSOD constants, sentinels, paths
│   ├── data_loader.py          anonymous S3 access, caching, failure handling
│   ├── preprocessing.py        sentinels, FRSHTT parsing, aggregation, scaling
│   ├── clustering.py           K-Means, elbow analysis, k selection, profiling
│   └── visualization.py        four output charts
├── tests/
│   ├── conftest.py             fixtures, fake S3 client, synthetic blobs
│   ├── test_data_loader.py     32 tests
│   ├── test_preprocessing.py   46 tests
│   ├── test_clustering.py      40 tests
│   └── test_integration.py     2 tests
├── data/                       download cache (git-ignored)
└── outputs/                    charts, CSVs, test log
```

---

## What the pipeline does

**1. Acquire.** Lists station objects under `s3://noaa-gsod-pds/{year}/` with a
paginator, samples from that pool at random under a fixed seed, then downloads
concurrently. Files are cached locally so repeated runs do not re-hit the
network, and individual download failures are logged and skipped rather than
aborting a run of several hundred files.

The random sampling is not cosmetic. S3 returns keys in lexicographic order and
GSOD station identifiers begin with the WMO block number, which is assigned
geographically — the low blocks are northern Europe, Scandinavia, Greenland and
Russia. Taking the first N keys returns an Arctic sample, not a global one. An
earlier version of this loader did exactly that, and every resulting cluster
fell between 59 and 71 degrees latitude with a silhouette of 0.31: K-Means was
asked to find climate types in a dataset containing only one climate, and
correctly reported that no good partition existed. Two regression tests now
assert that the sample is neither the first N keys nor confined to one end of
the key range.

**2. Preprocess.** Three things in this dataset break naive code:

- **Missing values are not blank.** GSOD writes `9999.9` for a missing
  temperature, `999.9` for missing wind speed, and `99.99` for missing
  precipitation. Pandas reads these as ordinary floats. Left in place, a
  station that stopped reporting for a month acquires a mean temperature in
  the thousands, and the clustering then separates stations by instrument
  uptime instead of by climate.
- **`FRSHTT` leading zeros are significant.** It is six binary digits — Fog,
  Rain, Snow, Hail, Thunder, Tornado — so `001000` means snow. Read as an
  integer it becomes `1000` and every flag position shifts. The loader reads
  all columns as text and the parser zero-pads before slicing.
- **`STP` is excluded by design.** NOAA documents the missing sentinel as
  `9999.9`, but `999.9` is both a plausible real station-pressure reading and
  the value actually written for missing data, so the column cannot be cleaned
  by sentinel matching without destroying genuine observations. Sea-level
  pressure carries the same signal without the defect.

Daily rows are then aggregated to one climate profile per station, stations
with fewer than 300 reporting days are dropped, residual gaps are median-filled,
and the ten features are standardised.

**3. Choose k.** Three signals, in order:

| Signal | Role |
|---|---|
| Inertia (within-cluster sum of squares) | Falls monotonically with k, so it can never select k alone. It only defines the elbow |
| **Davies-Bouldin** | Resolves the choice within the elbow's candidate range; lower is better |
| Silhouette | Confirms the partition; above 0.5 is treated as strong |
| Calinski-Harabasz | Third internal check |

The elbow is located geometrically as the point of maximum perpendicular
distance from the chord joining the first and last points of the inertia
curve, so the choice is reproducible rather than read off by eye. Where the
indices disagree, `select_optimal_k` reports the disagreement instead of
hiding it.

**4. Fit, profile, and plot.** Outputs land in `outputs/`.

---

## Outputs

| File | Contents |
|---|---|
| `elbow_analysis.png` | Four-panel comparison of the quality indices across k |
| `cluster_projection.png` | Clusters projected onto two principal components |
| `geographic_distribution.png` | Stations by longitude and latitude, coloured by cluster |
| `cluster_profiles.png` | Heatmap of cluster means for naming the groups |
| `clustered_stations.csv` | Every station with its assigned cluster |
| `cluster_summary.csv` | Mean feature values per cluster |
| `elbow_metrics.csv` | The full index table |
| `unit_test_results.txt` | pytest log |

---

## Testing

```bash
pytest                                   # 120 tests
pytest --cov=src --cov-report=term-missing
pytest tests/test_clustering.py -v       # one module
```

No test touches the network. S3 access is exercised through a fake client in
`conftest.py`, so the suite is deterministic and runs offline.

The suite covers input validation, edge cases (k=1, k=n, identical points,
NaN, infinity, single feature), caching behaviour, partial download failure,
sampling reproducibility and spread, and each documented sentinel value
individually.

**Clustering correctness** is verified against ground truth rather than
assumed. `TestClusteringCorrectness` builds three well-separated Gaussian
blobs with known membership and asserts that K-Means recovers the original
grouping with an Adjusted Rand Index above 0.95, that the recovered centroids
sit near the true centres, and that the elbow analysis independently selects
the correct k. Reported inertia is cross-checked against a hand-computed
within-cluster sum of squares.

Visualisation code is intentionally not unit-tested; chart rendering is
verified by inspecting the output files.

---

## A note on the projection charts

Clustering is performed in the full ten-dimensional standardised feature
space. The PCA scatter is a two-dimensional shadow of that partition for
inspection only — points that appear adjacent in the projection are not
necessarily neighbours in the space where the clusters were actually formed.

Latitude and longitude are **not** model inputs. They appear only in
`geographic_distribution.png`, which is therefore an independent check: if
clusters formed from weather variables alone fall into coherent latitude
bands, the algorithm has recovered real climate structure rather than noise.

---

## Data citation

NOAA National Centers for Environmental Information. *Global Surface Summary
of the Day (GSOD)*. Accessed via the Registry of Open Data on AWS,
`s3://noaa-gsod-pds`. https://registry.opendata.aws/noaa-gsod/

NOAA data disseminated through the NOAA Open Data Dissemination programme are
open to the public with no use restrictions. NOAA requests attribution and
does not endorse derived work.

## Licence

Course assignment. NOAA data is public domain.
