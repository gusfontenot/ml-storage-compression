- Status: preliminary results.
- Environment: local Ubuntu 24.04, WSL2.
- Data: nine generated workloads, 2,160 measured codec runs.
- CPU quotas: 25% and 50%.
- Chunk size: 1 MiB. CPU period: 100 ms.
- Training: 560 decisions from seven workloads.
- Validation: 80 decisions from one separate workload, including both quotas and all repeats.
- Final test: reserved.

- No compression: 0% storage savings, 12.74 ms average write time.
- LZ4: 24.5% storage savings, 11.13 ms average write time.
- Zstandard: 24.9% storage savings, 16.49 ms average write time.
- ML selector: 24.9% storage savings, 16.49 ms average write time in offline replay.
- Write time: compression through fsync. Replay excludes live selector costs.

- LZ4 size prediction error: 256,542 bytes before, 131,032 bytes after, 48.9% lower.
- Zstandard size prediction error: 260,818 bytes before, 133,962 bytes after, 48.6% lower.
- Error measure: mean absolute error on the same local validation data.
- Original model: VM-trained. New model: locally trained with more workload variety.

- ML selected Zstandard for all 80 validation decisions.
- Fixed LZ4 was faster than ML replay on this workload.
- Live integration: 128 readback checks passed with the original model.
- Example 1 ms budget: missed by every validation write.
- Next improvement: timing predictions and codec choices.

- Validation details: [evaluation.json](../models/local_v2_validation/evaluation.json).
- Live check: [local_live_01.json](../experiments/local_live_01.json).
- Run history: [runs.csv](../experiments/runs.csv).
