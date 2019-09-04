from setuptools import setup

setup(
    name='nwae',
    version='0.3.0',
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
        'nwae.lib.lang.stats'
    ],
    package_dir={'': 'src'},
    install_requires = [
        'nwae.utils',
        'tensorflow',
        'Keras'
    ],
    url='',
    license='',
    author='NWAE',
    author_email='705270564@qq.com',
    description='ML'
)
