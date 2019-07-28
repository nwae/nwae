from setuptools import setup

setup(
    name='mozg',
    version='0.1.0',
    packages=[
        'mozg.lib.math', 'mozg.lib.math.ml', 'mozg.lib.math.ml.metricspace'
    ],
    package_dir={'': 'src'},
    url='',
    license='',
    author='mark.tan',
    author_email='mapktah@gmail.com',
    description='ML Algos'
)
