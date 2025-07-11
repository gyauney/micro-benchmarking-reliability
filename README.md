## reliable-microbenchmarking

This repository contains code and data for reproducing the results from 
"How Reliable is Language Model Micro-benchmarking?"

**Download cached micro-benchmarking results:**
[Download from Google Drive (350 MB)](https://drive.google.com/file/d/1zv-SJdJQYpOjjlRbDxZVFacShD5M5oGg/view?usp=sharing) and unzip into this directory.

**Install requirements:**

```
pip install -r requirements.txt
```

## Steps to reproduce the results in the paper

You can skip the first step if you have downloaded the cached micro-benchmarking
results above.

### 1. Run micro-benchmarking methods:

First, download the cached model evaluation results from the
[Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/blog):
[download from Google Drive (550 MB)](https://drive.google.com/file/d/1OG1KJEyCTPbfo9KeB4b9wjqeXagc1DHK/view?usp=sharing) and unzip into this directory.

Here is an example command to run the micro-benchmarking evaluations for the
MMLU-Pro dataset:

```
python evaluate-microbenchmarks.py --selection_techniques Random Random_Subtask_Stratified_Equal Anchor_Points_Weighted Stratified_Random_Sampling tinyBenchmarks DPP --num_source_models 300 --num_runs 50 --benchmark mmlu-pro --combine_subtasks --same_points --num_threads 10
```

To reproduce the main results in the paper, you will need to run this for the
other benchmarks as well: `mmlu`, `bbh`, `gpqa`.

### 2. Process results:

All results need to be processed by running the following command:

```
python process-results-combine-subtasks.py
```

### 3. Make all plots:

Each file that begins with `figure` can be used to reproduce a figure from the paper.
For example, `figure-1.py` will reproduce Figure 1.

### Licenses

This code is released under the MIT License.
We use and adapt code from [Anchor Points](https://github.com/rvivek3/AnchorPoints),
[tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks),
[py-irt](https://github.com/nd-ball/py-irt),
and [DPPcoresets](https://github.com/hsimonfroy/DPPcoresets).
Their licenses are available in the `licenses` directory.