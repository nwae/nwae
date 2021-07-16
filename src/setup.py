from setuptools import setup

setup(
    name='nwae',
    version='1.7.0',
    packages=[
        # 'nwae.config',
        'nwae.lib.lang',
        'nwae.lib.lang.characters',
        'nwae.lib.lang.classification',
        'nwae.lib.lang.detect',
        'nwae.lib.lang.detect.comwords',
        'nwae.lib.lang.model',
        'nwae.lib.lang.nlp',
        'nwae.lib.lang.nlp.daehua',
        'nwae.lib.lang.nlp.daehua.forms',
        'nwae.lib.lang.nlp.lemma',
        'nwae.lib.lang.nlp.sajun',
        'nwae.lib.lang.nlp.translation',
        'nwae.lib.lang.nlp.ut',
        'nwae.lib.lang.preprocessing',
        'nwae.lib.lang.preprocessing.ut',
        'nwae.lib.lang.stats',
        'nwae.lib.lang.tts',
        'nwae.lib.lang.tts.examples',
        'nwae.lib.math',
        'nwae.lib.math.ml',
        'nwae.lib.math.ml.deeplearning',
        'nwae.lib.math.ml.examples',
        'nwae.lib.math.ml.metricspace',
        'nwae.lib.math.ml.metricspace.ut',
        'nwae.lib.math.ml.sequence',
        'nwae.lib.math.optimization',
        # 'nwae.samples',
        # 'nwae.ut',
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
