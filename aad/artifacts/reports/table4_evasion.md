# Table IV - evasion of the baseline NIDS under white-box attack

Source rows: 5,000 true-attack flows drawn from `data/processed/test.npz` with seed 42. The **same rows** are used for every cell, so the twelve results are directly comparable.

`clean_acc` is the model's detection rate on those rows before perturbation; `adv_acc` is the same rows after. `evasion` is the fraction of the *correctly-classified* rows that the attack flipped to benign -- the honest denominator. Precision and FPR are undefined on an all-attack subset and are deliberately omitted.

| model   | attack   | eps   |   clean_acc_% |   adv_acc_% |   evasion_% |   linf_max |   l2_mean |   feats_changed |   seconds |
|:--------|:---------|:------|--------------:|------------:|------------:|-----------:|----------:|----------------:|----------:|
| MLP     | FGSM     | 0.1   |         99.88 |        0    |     100     |    0.1     |    0.7118 |            57   |       0.1 |
| MLP     | BIM      | 0.1   |         99.88 |        0    |     100     |    0.1     |    0.5769 |            46.4 |       0.7 |
| MLP     | PGD      | 0.1   |         99.88 |        0    |     100     |    0.1     |    0.6229 |            46.3 |       3.2 |
| MLP     | DEEPFOOL | -     |         99.88 |        0.12 |     100     |    0.15883 |    0.0434 |            57   |       2.2 |
| CNN     | FGSM     | 0.1   |         99.76 |        0    |     100     |    0.1     |    0.5117 |            29.7 |       0.4 |
| CNN     | BIM      | 0.1   |         99.76 |        0    |     100     |    0.1     |    0.3693 |            43.5 |       4.4 |
| CNN     | PGD      | 0.1   |         99.76 |        0    |     100     |    0.1     |    0.4198 |            45.8 |      17.4 |
| CNN     | DEEPFOOL | -     |         99.76 |        0.24 |      99.98  |    0.07622 |    0.0357 |            30   |      52.5 |
| LSTM    | FGSM     | 0.1   |         98.54 |        0    |     100     |    0.1     |    0.5875 |            45.2 |       2.5 |
| LSTM    | BIM      | 0.1   |         98.54 |        0    |     100     |    0.1     |    0.3055 |            45.6 |      27.2 |
| LSTM    | PGD      | 0.1   |         98.54 |        0    |     100     |    0.1     |    0.3452 |            43   |     108.2 |
| LSTM    | DEEPFOOL | -     |         98.54 |        1.78 |      99.675 |    0.93888 |    0.0975 |            45.3 |     512   |

## Paper comparison

Verma et al. report a single unattributed collapse figure per model. Ours is reported per attack, which is why there are twelve numbers here and three there.

| Model | Paper (under attack) | This run (worst attack) |
| --- | ---: | ---: |
| MLP | 24.95% | 0.00% |
| CNN | 49.76% | 0.00% |
| LSTM | 4.89% | 0.00% |

## Accuracy vs. perturbation budget

Computed on a fixed 1,000-row subset of the same source rows. DeepFool is absent because its `epsilon` is an overshoot multiplier, not an L-inf budget -- sweeping it would not mean anything.

|                  |   0.001 |   0.002 |   0.005 |   0.01 |   0.02 |   0.05 |   0.1 |   0.3 |
|:-----------------|--------:|--------:|--------:|-------:|-------:|-------:|------:|------:|
| ('cnn', 'bim')   |    99.9 |    99.9 |    99.8 |   91.1 |    4.9 |      0 |     0 |     0 |
| ('cnn', 'fgsm')  |    99.9 |    99.9 |    99.8 |   91.7 |    5   |      0 |     0 |     0 |
| ('cnn', 'pgd')   |    99.9 |    99.9 |    99.8 |   91.1 |    4.9 |      0 |     0 |     0 |
| ('lstm', 'bim')  |    98.2 |    98.2 |    98.1 |   84   |    0   |      0 |     0 |     0 |
| ('lstm', 'fgsm') |    98.2 |    98.2 |    98.2 |   92.9 |    1.8 |      0 |     0 |     0 |
| ('lstm', 'pgd')  |    98.2 |    98.2 |    98.1 |   84   |    0   |      0 |     0 |     0 |
| ('mlp', 'bim')   |    99.9 |    99.9 |     7.9 |    2.4 |    0   |      0 |     0 |     0 |
| ('mlp', 'fgsm')  |    99.9 |    99.9 |     8   |    2.6 |    0   |      0 |     0 |     0 |
| ('mlp', 'pgd')   |    99.9 |    99.9 |     7.9 |    2.4 |    0   |      0 |     0 |     0 |

Detection rate (%). Accuracy must fall as eps rises; a non-monotone row is evidence of gradient masking and is itself a reportable finding.
