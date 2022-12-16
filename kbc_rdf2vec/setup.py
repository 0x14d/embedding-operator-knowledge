from setuptools import setup

setup(
    name="kbc_rdf2vec",
    version="0.1",
    packages=["kbc_rdf2vec"],
    url="www.jan-portisch.eu",
    license="MIT",
    author="Jan Portisch",
    author_email="jan@informatik.uni-mannheim.de",
    description="Allows to use RDF2Vec for knowledge base completion (KBC) / link predictions.",
    package_data={"kbc_rdf2vec": ["log.conf", "datasets/fb15k/*", "datasets/wn18/*"]},
)
