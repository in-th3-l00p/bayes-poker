# Reading Folded Hands: Opponent Modelling in Heads-Up Limit Hold'em

A Bayesian network that guesses an opponent's hidden cards and playing style
in two-player limit Texas Hold'em, learned from public hand histories.

Paper (compiled): [paper/paper.pdf](paper/paper.pdf)

## Contents

- `paper/` - the paper (LaTeX source, figures, compiled PDF)
- `implementation/` - the model (`holdem_model.py`), the notebook with all
  experiments (`notebook.ipynb`), a PettingZoo demo and the data scripts
- `submission/` - the course project on naive and non-naive Bayes classifiers:
  a 2 page report (`raport.pdf`, in Romanian) and a single script
  (`bayes_poker.py`) that downloads the data and reproduces every number
- `draft/` - an earlier, longer draft of the paper

## Running

The submission script needs Python 3.9+, numpy and matplotlib:

```
cd submission
python3 bayes_poker.py
```

The implementation uses the environment in `implementation/devbox.json` and
`implementation/requirements.txt`. Data is fetched with
`implementation/data/fetch.sh`.

## Data

Hand histories come from the PHH dataset on Zenodo
(DOI 10.5281/zenodo.10796885), including matches from the Annual Computer
Poker Competition.
