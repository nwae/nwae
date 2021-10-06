from setuptools import setup

setup(
    name='nwae',
    version='1.8.9',
    packages=[
        'nwae.lang',
        'nwae.lang.characters',
        'nwae.lang.classification',
        'nwae.lang.config',
        'nwae.lang.corpora',
        'nwae.lang.detect',
        'nwae.lang.detect.comwords',
        'nwae.lang.model',
        'nwae.lang.nlp',
        'nwae.lang.nlp.daehua',
        'nwae.lang.nlp.daehua.forms',
        'nwae.lang.nlp.lemma',
        'nwae.lang.nlp.sajun',
        'nwae.lang.nlp.translation',
        'nwae.lang.nlp.ut',
        'nwae.lang.preprocessing',
        'nwae.lang.preprocessing.ut',
        'nwae.lang.speech',
        'nwae.lang.speech.recognition',
        'nwae.lang.speech.tts',
        'nwae.lang.stats',
        'nwae.math',
        'nwae.math.config',
        'nwae.math.data',
        'nwae.math.fit',
        'nwae.math.fit.markov',
        'nwae.math.measures',
        'nwae.math.nn',
        'nwae.math.nn.loss',
        'nwae.math.number',
        'nwae.math.optimization',
        'nwae.math.suggest',
        'nwae.math.tree',
        'nwae.ml',
        'nwae.ml.boosting',
        'nwae.ml.config',
        'nwae.ml.data',
        'nwae.ml.decisiontree',
        'nwae.ml.metricspace',
        'nwae.ml.metricspace.ut',
        'nwae.ml.modelhelper',
        'nwae.ml.networkdesign',
        'nwae.ml.nndense',
        'nwae.ml.sequence',
        'nwae.ml.text',
        'nwae.ml.text.preprocessing',
        'nwae.ml.trainer',
        # 'nwae.samples',
        # 'nwae.ut',
    ],
    package_dir={'': 'src'},
    install_requires = [
        'nwae.utils',
        'mex',
        'nltk',
    ],
    url='',
    license='',
    author='NWAE',
    author_email='mapktah@ya.ru',
    description='ML'
)
