# Table III - NIDS accuracy and loss (clean test set)

Data: `data/processed` | test rows: 224,469 (21,870 attack, 202,599 benign)
Seed 42, batch 1024, Adam lr=0.001, early stopping on val_loss (patience 4).

| model   |   accuracy_% |   precision_% |   recall_% |    f1_% |   val_loss |   fpr_% |   fn |   fp |   params |   epochs |   train_s |
|:--------|-------------:|--------------:|-----------:|--------:|-----------:|--------:|-----:|-----:|---------:|---------:|----------:|
| MLP     |      99.9898 |       99.9817 |    99.9131 | 99.9474 |    0.00054 |  0.002  |   19 |    4 |    19234 |       20 |      53.8 |
| CNN     |      99.9795 |       99.9817 |    99.808  | 99.8947 |    0.00087 |  0.002  |   42 |    4 |    33346 |       30 |    1237.3 |
| LSTM    |      99.8387 |       99.8055 |    98.5368 | 99.1671 |    0.0093  |  0.0207 |  320 |   42 |    19042 |       13 |    2486.3 |

## Detection rate per attack type

| class                  |   support |     MLP |     CNN |   LSTM |
|:-----------------------|----------:|--------:|--------:|-------:|
| Benign                 |    202599 |  99.998 |  99.998 | 99.979 |
| Type1 DoS-Hulk         |     21780 |  99.995 | 100     | 98.944 |
| Type2 DoS-SlowHTTPTest |         9 | 100     | 100     |  0     |
| Type3 BruteForce-Web   |        51 |  66.667 |  49.02  |  0     |
| Type4 BruteForce-XSS   |        23 |  95.652 |  39.13  |  0     |
| Type5 SQL-Injection    |         7 | 100     |  71.429 |  0     |
