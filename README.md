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

This Repository contains the code necessary to reproduce the results shown in the paper, which was submitted to DL4KG.

## How to execute the results

### Generating the embeddings:

In order to generate the embeddings execute:

`python -m knowledge_infusion.compare_methods.compare_methods`

Be cautious, this will take some time.

### Generating the tables from the results 

After generating the embeddings execute:

`python -m knowledge_infusion.compare_methods.generate_output_files`

This will generate the resulting tables for:
- Each iteration in [knowledge_infusion/compare_methods/results/iteration*/_table_format/]()
- all iterations combined in [knowledge_infusion/compare_methods/results/_table_format/]()
