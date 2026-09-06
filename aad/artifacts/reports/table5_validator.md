# Table V - validator catch-rate and false-positive rate

Rules mined from 945,461 **benign training** rows. The distribution threshold is calibrated on benign *validation* rows at a 1.0% budget. Everything below is measured on **test**, which neither pass saw.

Catch-rate and FPR are reported separately and per rule family. A validator rejecting 3% of legitimate traffic is unusable whatever its catch rate, and a single blended number would hide that.

| set                 | kind   |      n |   combined_% |   range_% |   integrality_% |   dependency_% |   distribution_% |   clean_orig_% |   attributable_% |
|:--------------------|:-------|-------:|-------------:|----------:|----------------:|---------------:|-----------------:|---------------:|-----------------:|
| clean benign (test) | FPR    | 202599 |        0.948 |     0     |               0 |          0.001 |            0.947 |         nan    |           nan    |
| clean attack (test) | reject |  21870 |        0.091 |     0.014 |               0 |          0     |            0.078 |         nan    |           nan    |
| bim x cnn           | catch  |   5000 |      100     |     0.02  |             100 |        100     |           17.58  |           0.12 |            99.88 |
| bim x lstm          | catch  |   5000 |      100     |     0.02  |             100 |        100     |           99.86  |           0.12 |            99.88 |
| bim x mlp           | catch  |   5000 |      100     |     0     |             100 |        100     |          100     |           0.12 |            99.88 |
| deepfool x cnn      | catch  |   5000 |      100     |     0.02  |             100 |        100     |            1     |           0.12 |            99.88 |
| deepfool x lstm     | catch  |   5000 |      100     |     0.02  |             100 |        100     |            2.28  |           0.12 |            99.88 |
| deepfool x mlp      | catch  |   5000 |      100     |     0.02  |             100 |        100     |            1.18  |           0.12 |            99.88 |
| fgsm x cnn          | catch  |   5000 |      100     |     0.02  |             100 |        100     |          100     |           0.12 |            99.88 |
| fgsm x lstm         | catch  |   5000 |      100     |     0.02  |             100 |        100     |          100     |           0.12 |            99.88 |
| fgsm x mlp          | catch  |   5000 |      100     |     0     |             100 |        100     |          100     |           0.12 |            99.88 |
| pgd x cnn           | catch  |   5000 |      100     |     0.02  |             100 |        100     |           94.98  |           0.12 |            99.88 |
| pgd x lstm          | catch  |   5000 |      100     |     0.02  |             100 |        100     |           99.88  |           0.12 |            99.88 |
| pgd x mlp           | catch  |   5000 |      100     |     0     |             100 |        100     |          100     |           0.12 |            99.88 |

## Reading this

- **FPR** on clean benign test traffic is the cost of deploying the gate: 0.948% of legitimate flows dropped. Note the 1.0% budget is spent by the *distribution* family alone; the four families are OR-ed, so the combined FPR is roughly their sum. Tighten `--fpr-budget` if the combined figure has to sit under a hard limit.
- **reject** on clean *attack* traffic is not an error -- those are real intrusions and the EIDS behind the gate would have flagged them anyway. It is reported so the gate's behaviour is fully described.
- **catch** is the fraction of adversarial rows the gate stops before any model sees them.
- **clean_orig** is the gate's reject rate on the *same rows unperturbed*, and **attributable** is the fraction it catches that it would not have caught anyway. This distinction is easy to miss and it matters: every adversarial row here was built from a real attack flow, and real attack flows already fall outside the benign envelope the rules were mined from. A headline catch rate therefore partly measures *maliciousness* rather than *perturbation*. **`attributable` is the honest measure of what the gate adds as an adversarial-example detector**; read the two together or the result is overstated.

The **integrality** family alone accounts for most of the catch rate (100.00% mean across the twelve cells). That is the honest finding: the defense rests on arithmetic that a gradient attack violates incidentally, not on anything the attacker is forced to violate. An adaptive adversary that rounds its perturbation onto the integer lattice and re-derives the dependent features would pass this gate untouched -- which is what phase 8 is for.
