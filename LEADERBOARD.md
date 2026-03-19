# DanceTrack Leaderboard

Published methods evaluated on the [DanceTrack](https://dancetrack.github.io/) test set. Results are from the [CodaBench evaluation server](https://www.codabench.org/competitions/14885/#/results-tab).

For a **sortable table** (click any column header to rank), see [`LEADERBOARD.csv`](LEADERBOARD.csv). GitHub renders CSV files with built-in column sorting.

## Test Set Results

Please add your result here according to the publication date order (the newer the later).

| Method | HOTA | DetA | AssA | MOTA | IDF1 | CodaBench ID | Paper | Venue | Date |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|
| [OC-SORT](https://github.com/noahcao/OC_SORT) | 55.7 | 81.7 | 38.1 | 92.0 | 54.6 | [628670](https://www.codabench.org/competitions/14885/#/submission/628670) | [arXiv](https://arxiv.org/abs/2203.14360) | CVPR 2023 | 2023-06 |

## Metrics

- **HOTA** (Higher Order Tracking Accuracy): Primary ranking metric. Balances detection and association.
- **DetA** (Detection Accuracy): Measures detection quality.
- **AssA** (Association Accuracy): Measures identity association quality.
- **MOTA** (Multiple Object Tracking Accuracy): Classic metric from CLEAR MOT.
- **IDF1** (ID F1 Score): Measures identity preservation.

## Contributing

To add your method to this leaderboard:
1. Submit results to [CodaBench](https://www.codabench.org/competitions/14885/) and make it public.
2. Open a pull request (with "[LEADERBOARD]" in the title) adding a row to both `LEADERBOARD.md` and `LEADERBOARD.csv` with your method name, CodaBench submission ID, performance metrics, paper link, venue, and publication date.
