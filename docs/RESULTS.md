# Preliminary results

- Environment: local Ubuntu 24.04, WSL2.
- Data: nine generated workloads; 2,160 measured codec runs.
- CPU quotas: 25%, 50%. Chunk: 1 MiB. Period: 100 ms.
- Training: 560 decisions, seven workloads.
- Validation: 80 decisions, one held-out workload, both quotas and all repeats.
- Final test: reserved.

## Validation

| Policy | Storage savings | Mean write time |
| --- | ---: | ---: |
| No compression | 0.0% | 12.74 ms |
| LZ4 | 24.5% | 11.13 ms |
| Zstandard | 24.9% | 16.49 ms |
| ML selector (replay) | 24.9% | 16.49 ms |

Write time: compression through fsync. ML replay uses recorded codec measurements; live selector costs are excluded.

## Size prediction error

| Codec | Original model | New model | Reduction |
| --- | ---: | ---: | ---: |
| LZ4 | 256,542 bytes | 131,032 bytes | 48.9% |
| Zstandard | 260,818 bytes | 133,962 bytes | 48.6% |

Mean absolute error on the same local validation data. Original model: VM-trained. New model: locally trained with more workload variety.

## Findings

- Size predictions: about 49% lower error.
- ML choices: Zstandard for all 80 decisions.
- Storage: 24.9% savings on this validation workload.
- Speed: fixed LZ4 was faster than ML replay.
- Live integration: 128 readback checks passed with the original model.
- Example 1 ms budget: missed by every validation write.
- Status: working prototype; timing predictions need improvement.

Details: [validation](../models/local_v2_validation/evaluation.json), [live check](../experiments/local_live_01.json), [history](../experiments/runs.csv).
