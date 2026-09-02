# Table I - Statistical description of the dataset

Source: Friday-16-02-2018_TrafficForML_CICFlowMeter.csv, Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
Rows ingested: 2,097,150 -> retained after cleaning: 1,496,454
Features retained: 68 of 78
Split: 0.7/0.15/0.15 stratified on the 6-way label, seed 42

| paper_type   | class                    | binary   |   total |   train |    val |   test |   pct_of_total |
|:-------------|:-------------------------|:---------|--------:|--------:|-------:|-------:|---------------:|
| -            | Benign                   | benign   | 1350659 |  945461 | 202599 | 202599 |        90.2573 |
| Type1        | DoS attacks-Hulk         | attack   |  145199 |  101639 |  21780 |  21780 |         9.7029 |
| Type3        | Brute Force -Web         | attack   |     340 |     238 |     51 |     51 |         0.0227 |
| Type4        | Brute Force -XSS         | attack   |     150 |     105 |     22 |     23 |         0.01   |
| Type2        | DoS attacks-SlowHTTPTest | attack   |      55 |      38 |      8 |      9 |         0.0037 |
| Type5        | SQL Injection            | attack   |      51 |      36 |      8 |      7 |         0.0034 |
|              | ALL                      |          | 1496454 | 1047517 | 224468 | 224469 |       100      |

## Cleaning ledger

- Embedded header rows removed: 1
- Non-finite (Infinity) cells replaced: 7,662 in ['Flow Byts/s', 'Flow Pkts/s']
- Rows dropped for NaN: 5,708
- Exact duplicate rows dropped: 594,987
- Zero-variance columns dropped (detected on train only): ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']
