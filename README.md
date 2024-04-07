# hyperparameter-sensitivity
Testing hyperparameter sensitivity between LSTM/Transformer in character transduction tasks.

## Usage
Requires yoyodyne, which is dependent on python>=3.9. Currently we make use of commit cc4e991. Install with:

```
mkdir lib
cd lib
git clone git@github.com:CUNY-CL/yoyodyne.git
cd yoyodyne
git reset cc4e991 --hard
pip install .
```

## Directory
- For logs of all of the sweeps, see `results`
- In `experiments` we have slurm scripts for queuing training jobs
- In `scripts`, we have the training script, as well as well as some additional scripts for getting test results from the best dev run in each sweep, and additional utilities to generate W&B sweeps, etc.

## Citation
If you find this work useful, you can cite

```
@inproceedings{wiemerslage-etal-2024-quantifying,
    title = "Quantifying the Hyperparameter Sensitivity of Neural Networks for Character-level Sequence-to-Sequence Tasks",
    author = "Wiemerslage, Adam  and
      Gorman, Kyle  and
      Wense, Katharina",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.40",
    pages = "674--689",
}
```


