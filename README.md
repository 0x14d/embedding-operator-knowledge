# A Closer Look at Sum-based Embeddings for Knowledge Graphs Containing Procedural Knowledge

**This repository will be refactored in the following days**

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
