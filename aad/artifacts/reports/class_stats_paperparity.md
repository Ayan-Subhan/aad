# Table I - Statistical description of the dataset

Source: Friday-16-02-2018_TrafficForML_CICFlowMeter.csv, Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
Rows ingested: 2,097,150 -> retained after cleaning: 2,091,441
Features retained: 68 of 78
Split: 0.7/0.15/0.15 stratified on the 6-way label, seed 42

| paper_type   | class                    | binary   |   total |   train |    val |   test |   pct_of_total |
|:-------------|:-------------------------|:---------|--------:|--------:|-------:|-------:|---------------:|
| -            | Benign                   | benign   | 1489073 | 1042351 | 223361 | 223361 |        71.1984 |
| Type1        | DoS attacks-Hulk         | attack   |  461912 |  323338 |  69287 |  69287 |        22.0858 |
| Type2        | DoS attacks-SlowHTTPTest | attack   |  139890 |   97923 |  20983 |  20984 |         6.6887 |
| Type3        | Brute Force -Web         | attack   |     362 |     253 |     54 |     55 |         0.0173 |
| Type4        | Brute Force -XSS         | attack   |     151 |     106 |     23 |     22 |         0.0072 |
| Type5        | SQL Injection            | attack   |      53 |      37 |      8 |      8 |         0.0025 |
|              | ALL                      |          | 2091441 | 1464008 | 313716 | 313717 |       100      |

## Cleaning ledger

- Embedded header rows removed: 1
- Non-finite (Infinity) cells replaced: 7,662 in ['Flow Byts/s', 'Flow Pkts/s']
- Rows dropped for NaN: 5,708
- Exact duplicate rows dropped: 0
- Zero-variance columns dropped (detected on train only): ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']
