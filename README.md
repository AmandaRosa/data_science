# **Data Science Package**
[![Language: python](https://img.shields.io/badge/Language-python-blue)]()
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://github.com/ambv/black)
[![OS: linux](https://img.shields.io/badge/OS-Linux-yellow)]()

<div align="justify">
The Data Science is an under development software tool used in the EDGE Analytics, a project of Petrobras in partnership with the Structural Mechanics Laboratory (LMEst) of the Federal University of Uberlândia. The main goal of this project is to reduce the size of the company's database by using artificial intelligence (AI). This package contains the developed AI classes.

The storage optimization is achieved through the classification of the signals from rotating machines, such as turbo compressors, pumps, turbo generators, and other monitored devices. The classification is based on the identification of anomalies, i.e., if the signal has or not an odd behavior when compared to an healthy signal build by the AI. Hence, if a signal has an anomaly, it will be saved in the database, else it will not.
</div>

## **Install the Data Science**

Use the following command to use the package:

```sh
pip install .
```

But if you want to use it as developer, i.e., modify the documentation, format code using black, and other things, run the command bellow:

```sh
pip install -e .[dev]
```

## **Documentation**

The documentation site can be found [here](https://edge-analytics.gitlab.io/artificial-intelligence/data_science).