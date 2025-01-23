# process-reward-model-reproduce
---
Reproduce the Process Reward Model (PRM) from OpenAI's paper 《Let's verify step by step》


## Notice ! This project is still under development. Please wait a few weeks.


## Main Contributions of the paper.
---

1. **Superior Process Supervision Performance**
- Demonstrated that process supervision significantly outperforms outcome supervision for training reliable reward models
- Achieved 78.2% solve rate on a representative subset of the MATH test set using their state-of-the-art Process-supervised Reward Model (PRM)

1. **Efficient Synthetic Supervision Method**
- Showed that a large reward model can reliably approximate human supervision for training smaller reward models
- This enabled efficient large-scale data collection ablations without requiring costly human feedback

1. **Active Learning Benefits**
- Demonstrated that active learning leads to a 2.6× improvement in the data efficiency of process supervision
- This significantly reduced the cost of human data collection by surfacing the most valuable model completions for feedback

1. **Dataset Release**
- Released PRM800K, a comprehensive dataset containing 800,000 step-level human feedback labels
- Dataset covers 75K solutions across 12K problems
- Made publicly available to promote and facilitate related research

1. **Alignment Implications**
- Showed that process supervision has positive alignment implications without performance trade-offs
- Demonstrated that process supervision actually provides better performance while being more interpretable and safer
- This "negative alignment tax" could encourage broader adoption of safer AI development practices






## Requirements
---
```bash


```





## How to run ?
1. download the `PRM800K` dataset
```python
from datasets import load_dataset  
dataset = load_dataset("openai/prm800k")
```
or
```bash

```

2. run the `main.py`
```bash
python main.py
```





## Citation
---
```bibtxt
@misc{lightman2023letsverifystepstep,
      title={Let's Verify Step by Step}, 
      author={Hunter Lightman and Vineet Kosaraju and Yura Burda and Harri Edwards and Bowen Baker and Teddy Lee and Jan Leike and John Schulman and Ilya Sutskever and Karl Cobbe},
      year={2023},
      eprint={2305.20050},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.20050}, 
}

```