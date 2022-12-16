from setuptools import setup

setup(
    name="kbc_evaluation",
    version="0.1",
    packages=["kbc_evaluation"],
    url="www.jan-portisch.eu",
    license="MIT",
    author="Jan Portisch",
    author_email="jan@informatik.uni-mannheim.de",
    description="Allows to evaluate link prediction text files.",
    package_data={
        "kbc_evaluation": ["log.conf", "datasets/fb15k/*", "datasets/wn18/*"]
    },
)
