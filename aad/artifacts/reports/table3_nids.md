# Table III - NIDS accuracy and loss (clean test set)

Data: `data/processed` | test rows: 224,469 (21,870 attack, 202,599 benign)
Seed 42, batch 1024, Adam lr=0.001, early stopping on val_loss (patience 4).

| model   |   accuracy_% |   precision_% |   recall_% |    f1_% |   val_loss |   fpr_% |   fn |   fp |   params |   epochs |   train_s |
|:--------|-------------:|--------------:|-----------:|--------:|-----------:|--------:|-----:|-----:|---------:|---------:|----------:|
| MLP     |      99.9898 |       99.9817 |    99.9131 | 99.9474 |    0.00055 |  0.002  |   19 |    4 |    19234 |       20 |     162.1 |
| CNN     |      99.9791 |       99.9771 |    99.808  | 99.8925 |    0.0008  |  0.0025 |   42 |    5 |    33346 |       30 |    2541.2 |
| LSTM    |      99.6997 |       97.5588 |    99.4056 | 98.4735 |    0.01702 |  0.2685 |  130 |  544 |    19042 |       30 |    7166.3 |

## Detection rate per attack type

| class                  |   support |     MLP |     CNN |    LSTM |
|:-----------------------|----------:|--------:|--------:|--------:|
| Benign                 |    202599 |  99.998 |  99.998 |  99.731 |
| Type1 DoS-Hulk         |     21780 |  99.995 | 100     |  99.775 |
| Type2 DoS-SlowHTTPTest |         9 | 100     | 100     | 100     |
| Type3 BruteForce-Web   |        51 |  66.667 |  49.02  |   0     |
| Type4 BruteForce-XSS   |        23 |  95.652 |  39.13  |   0     |
| Type5 SQL-Injection    |         7 | 100     |  71.429 |   0     |
