# KBC RDF2Vec
[![Python CI](https://github.com/janothan/kbc_rdf2vec/workflows/Python%20CI/badge.svg)](https://github.com/janothan/kbc_rdf2vec/actions) 
[![Coverage Status](https://coveralls.io/repos/github/janothan/kbc_rdf2vec/badge.svg)](https://coveralls.io/github/janothan/kbc_rdf2vec) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/janothan/kbc_rdf2vec)](./LICENSE)

A simple Python project to generate a knowledge base completion file for evaluation given a gensim model.
The file can then be evaluated using [KBC Evaluation](https://github.com/janothan/kbc_evaluation/).

## Evaluation File Format

```
<valid triple>
    Heads: <concepts space separated>
    Tails: <concepts space separated>
```


### Development Remarks
- Docstring format: <a href="https://numpy.org/doc/stable/docs/howto_document.html">NumPy/SciPy</a>
- Code formatting: <a href="https://github.com/psf/black">black</a>
