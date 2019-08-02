from setuptools import setup

setup(
    name='mozg',
    version='0.1.0',
    packages=[
        'mozg.lib.math',
        'mozg.lib.math.ml'
        # 'mozg.lib.math.ml.metricspace',
        #'mozg.lib.lang',
        #'mozg.lib.lang.characters',
        #'mozg.lib.lang.classification',
        #'mozg.lib.lang.model',
        #'mozg.lib.lang.nlp',
        #'mozg.lib.lang.stats'
    ],
    package_dir={'': 'src'},
    url='',
    license='',
    author='Mark',
    author_email='mapktah@yandex.ru',
    description='ML Algos'
)
