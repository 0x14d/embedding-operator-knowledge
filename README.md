# A Closer Look at Sum-based Embeddings for Knowledge Graphs Containing Procedural Knowledge

**Warning this repository is still highly WIP and will be hugely refactored in the following days**

This Repository contains the code necessary to reproduce the results shown in the paper, which was submitted to DL4KG.

## How to execute the results

### Generating the embeddings:

In order to generate the embeddings execute:

`python -m knowledge_infusion.compare_methods.compare_methods`

Be cautious, this will take a long period of time.

### Generating the tables from the embeddings

After generating the embeddings execute:

`python -m knowledge_infusion.compare_methods.generate_output_files`

This will generate the resulting tables for:
- Each iteration in [knowledge_infusion/compare_methods/results/iteration*/_table_format/]()
- all iterations combined in [knowledge_infusion/compare_methods/results/_table_format/]()