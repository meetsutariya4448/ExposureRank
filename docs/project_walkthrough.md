# One-minute explanation

I trained small recommendation models on historical observed video feedback and then froze them. I asked whether changing which outcomes are visible or which candidates are evaluated can alter the apparent winner. The full test gave MF nDCG0.8675 versus item rates0.8665, with an inconclusive paired interval. I separately ran13 mask conditions and6 candidate tasks,200 draws each, and checked linear inverse-probability correction. The benchmark is static and selected; completion is not satisfaction and simulated probabilities are not real exposure.

# Five-minute walkthrough

1. Define source/license and the held-out observed catalog.
2. Explain pair hash, train-only baseline rates, MF BCE update and neural comparison.
3. Show full_reference.csv and the two paired intervals.
4. Compare one mask condition with one independently sampled candidate task; explain HT linear variance and the denied hidden-outcome lookup.
5. Reproduce a headline from its table, discuss concentration/near ties and the clean rebuild evidence.

# Deep dive

Follow data.py → training.py → committed evaluation_lock → evaluation.py → simulation.py. Derive sigmoid(z)-y for one MF update, compute toy nDCG by hand, and enumerate HT masks in tests/test_statistics.py. A known observation probability cancels only a linear numerator; unseen IDCG is not known. Show user bootstrap, three seeds and conditional simulator randomness as separate evidence layers.
