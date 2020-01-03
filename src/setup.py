from setuptools import setup

setup(
    name='nwae',
    version='1.3.0',
    packages=[
        'nwae.lib.math',
        'nwae.lib.math.ml',
        'nwae.lib.math.ml.metricspace',
        'nwae.lib.math.ml.deeplearning',
        'nwae.lib.math.optimization',
        'nwae.lib.lang',
        'nwae.lib.lang.characters',
        'nwae.lib.lang.classification',
        'nwae.lib.lang.model',
        'nwae.lib.lang.nlp',
        'nwae.lib.lang.nlp.daehua',
        'nwae.lib.lang.nlp.translation',
        'nwae.lib.lang.nlp.sajun',
        'nwae.lib.lang.nlp.lemma',
        'nwae.lib.lang.preprocessing',
        'nwae.lib.lang.stats',
        'nwae.lib.lang.tts'
    ],
    package_dir={'': 'src'},
    install_requires = [
        'nwae.utils',
        'mex'
        'nltk',
        'googletrans'
    ],
    url='',
    license='',
    author='NWAE',
    author_email='mapktah@ya.ru',
    description='ML'
)
