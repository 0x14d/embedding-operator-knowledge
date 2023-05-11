# Embeddings of Knowledge Graphs Containing Procedural Knowledge

## Setup

1. Clone the repository with `git clone https://github.com/0x14d/embedding-operator-knowledge`
1. Open repository with `cd embedding-operator-knowledge`
1. <b>Python 3.9</b> is required (2 options)
   - Use the included dev container (requires <b>Docker</b>)
   - Install python manually
1. Create virtual environment (optional)
   1. `python -m venv ./venv`
   1. `source ./venv/bin/activate`
1. Install python3.9-dev with `sudo apt install python3.9-dev`
1. Install requirements with `pip install -r requirements.txt`


## How to:
### Generating the embeddings:

In order to train the embeddings & prepare the evaluation either for LinkPrediction (hits@k & AMRI) or Matches (matches@k) and the amount of iterations required execute:

`python -m knowledge_infusion.compare_methods.compare_methods --iterations 30 --config LinkPrediction --dp synthetic --rule-extraction FromEdge --knowledge-extraction groundtruth`
or
`python -m knowledge_infusion.compare_methods.compare_methods --iterations 30 --config Matches --dp synthetic --rule-extraction FromEdge --knowledge-extraction groundtruth`

### Generating the evaluations

After generating the embeddings execute:

`python -m knowledge_infusion.compare_methods.evaluation_linkpred`
for hits@k and AMRI or
`python -m knowledge_infusion.compare_methods.evaluation_matchesatk`
for matches@k

This will generate the resulting tables for:
- LinkPrediction in [knowledge_infusion/compare_methods/results/LinkPrediction/SYNTHETIC/FROM_EDGES/groundtruth/combined_eval_results_with_std]()
- Matches@k in [knowledge_infusion/compare_methods/results/LinkPrediction/SYNTHETIC/FROM_EDGES/groundtruth/_tables]()