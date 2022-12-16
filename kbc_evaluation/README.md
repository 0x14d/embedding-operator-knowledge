# KBC Evaluation
[![Python CI](https://github.com/janothan/kbc_evaluation/workflows/Python%20CI/badge.svg)](https://github.com/janothan/kbc_evaluation/actions) 
[![Coverage Status](https://coveralls.io/repos/github/janothan/kbc_evaluation/badge.svg)](https://coveralls.io/github/janothan/kbc_evaluation) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/janothan/kbc_evaluation)](./LICENSE) <br/>
A simple Python project to evaluate knowledge base completion (KBC). This module is implemented to evaluate [RDF2Vec KBC](https://github.com/janothan/kbc_rdf2vec) but can be used to evaluate any file following the prescribed format.

## Usage
- requires Python 3.6 or higher

## Evaluation File Format
The expected evaluation file must be a UTF-8 encoded text file which follows the given format below:
```
<valid triple>
    head: <concepts space separated>
    tail: <concepts space separated>
<valid triple>
    ...
```
Optionally, confidences can be given for each concept using the suffix `_{<confidence>}`.
Note that no ordering based on confidences is performed.

*Example of a valid file without confidences:*
```
A B C
	Heads: B C D F G A W X Y Z
	Tails: A B C D E F G H I J
D E F
	Heads: D E F G H I J K L M
	Tails: F G H I J K L M N O
```

*Example of a valid file with confidences:*
```
A B C
	Heads: B_{0.123} C_{0.123} D_{0.123} F_{0.123} G_{0.123} A_{0.123} W_{0.123} X_{0.123} Y_{0.123} Z_{0.123}
	Tails: A_{0.123} B_{0.123} C_{0.123} D_{0.123} E_{0.123} F_{0.123} G_{0.123} H_{0.123} I_{0.123} J_{0.123}
D E F
	Heads: D_{0.123} E_{0.123} F_{0.123} G_{0.123} H_{0.123} I_{0.123} J_{0.123} K_{0.123} L_{0.123} M_{0.123}
	Tails: F_{0.123} G_{0.123} H_{0.123} I_{0.123} J_{0.123} K_{0.123} L_{0.123} M_{0.123} N_{0.123} O_{0.123}```
```

### Development Remarks
- Docstring format: <a href="https://numpy.org/doc/stable/docs/howto_document.html">NumPy/SciPy</a>
- Code formatting: <a href="https://github.com/psf/black">black</a>