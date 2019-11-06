from setuptools import setup

setup(
    name='nwae',
    version='1.1.2',
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
        'nwae.lib.lang.stats'
    ],
    package_dir={'': 'src'},
    install_requires = [
        'nwae.utils',
        'tensorflow',
        'Keras',
        'nltk',
        'googletrans'
    ],
    url='',
    license='',
    author='NWAE',
    author_email='m5251@naver.com',
    description='ML'
)
