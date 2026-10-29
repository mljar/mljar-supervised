---
title: "Open Source Acknowledgments and Licenses"
description: "Meet the open source projects behind MLJAR AutoML, with algorithm credits, dependency licenses, model-weight terms, and links to upstream sources."
---

# Built on the shoulders of giants

MLJAR AutoML is built with tools created by many talented people. Their work
helps us train models, find useful features, explain predictions, and create reports.

**Thank you to the researchers, developers, and communities behind these projects.**
We also thank everyone who writes documentation, fixes bugs, and helps other users.

Below you can see which projects we use, what they do, and their licenses.
A license explains how software may be used, changed, and shared.
MLJAR's own code uses the [MIT license](https://github.com/mljar/mljar-supervised/blob/master/LICENSE).
The projects we use keep their own licenses.

## The tools behind our models

MLJAR trains and compares models using the libraries below. We credit both the
people who developed the methods and the people who built the software.

| Model or feature | Built with | Main license |
| --- | --- | --- |
| Baseline | scikit-learn `DummyClassifier` / `DummyRegressor` | BSD-3-Clause |
| Linear | scikit-learn logistic and linear regression | BSD-3-Clause |
| Decision Tree | scikit-learn | BSD-3-Clause |
| Random Forest | scikit-learn | BSD-3-Clause |
| Extra Trees | scikit-learn | BSD-3-Clause |
| Nearest Neighbors | scikit-learn | BSD-3-Clause |
| Neural Network | scikit-learn multilayer perceptron | BSD-3-Clause |
| Xgboost | XGBoost contributors; `xgboost-cpu` or `xgboost` | Apache-2.0 |
| LightGBM | LightGBM developers and Microsoft contributors | MIT |
| CatBoost | CatBoost contributors | Apache-2.0 |
| RealMLP | David Holzmüller and collaborators, through pytabkit | Apache-2.0 |
| TabM mini | TabM researchers and pytabkit contributors; adapted MLJAR training loop | Apache-2.0 |
| Kumo | NVIDIA Kumo Tabular through structured-data-models | Apache-2.0 for the library; saved model license below |
| TabICLv2 | Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan, and contributors; through structured-data-models | BSD-3-Clause for the model code; Apache-2.0 for SDM |
| Ensemble selection and stacking | MLJAR, using the models listed above | MIT for MLJAR code; each model keeps its license |
| Golden features and feature selection | MLJAR, using scientific Python and scikit-learn | MIT for MLJAR code; library licenses also apply |
| Feature discovery | Features Goldmine and its contributors | Apache-2.0 |

Visit these projects to learn more or find out how to cite their research:
[scikit-learn](https://scikit-learn.org/stable/about.html),
[XGBoost](https://github.com/dmlc/xgboost),
[LightGBM](https://github.com/lightgbm-org/LightGBM),
[CatBoost](https://github.com/catboost/catboost),
[pytabkit / RealMLP](https://github.com/dholzmueller/pytabkit),
[TabM](https://github.com/yandex-research/tabm),
[structured-data-models](https://github.com/NVIDIA/structured-data-models),
[TabICL](https://github.com/soda-inria/tabicl), and
[Features Goldmine](https://github.com/mljar/features_goldmine).
If you use these methods in a paper, please follow the projects' citation instructions.

## Packages installed with AutoML

These packages help AutoML work. Click a package name to visit its PyPI page.
Your computer uses either `xgboost-cpu` or `xgboost`, depending on the platform.

The license column gives the main license. Some packages include other projects
with their own license terms; those are marked below.

| Package | Used for | License summary |
| --- | --- | --- |
| [numpy](https://pypi.org/project/numpy/2.5.4/) | Working with numbers | BSD-3-Clause; includes other licenses too |
| [pandas](https://pypi.org/project/pandas/3.1.0/) | Working with tables of data | BSD-3-Clause |
| [scipy](https://pypi.org/project/scipy/1.18.1/) | Scientific computing and statistics | BSD-3-Clause; includes other licenses too |
| [scikit-learn](https://pypi.org/project/scikit-learn/1.9.1/) | Training and checking models; preparing data | BSD-3-Clause |
| [features-goldmine](https://pypi.org/project/features-goldmine/1.1.0/) | Finding useful new input features | Apache-2.0; see note below |
| [xgboost-cpu](https://pypi.org/project/xgboost-cpu/3.4.1/) | XGBoost for supported Linux and Windows computers | Apache-2.0 |
| [xgboost](https://pypi.org/project/xgboost/3.4.1/) | XGBoost on other platforms | Apache-2.0 |
| [lightgbm](https://pypi.org/project/lightgbm/4.7.0/) | Gradient-boosted trees | MIT; includes other licenses too |
| [catboost](https://pypi.org/project/catboost/1.2.10/) | Gradient-boosted trees | Apache-2.0 |
| [joblib](https://pypi.org/project/joblib/1.6.0/) | Saving models and running work in parallel | BSD-3-Clause |
| [tabulate](https://pypi.org/project/tabulate/0.10.0/) | Report tables | MIT |
| [matplotlib](https://pypi.org/project/matplotlib/3.11.2/) | Plots and figures | Matplotlib license; includes other licenses too |
| [dtreeviz](https://pypi.org/project/dtreeviz/2.3.2/) | Drawing decision trees | MIT |
| [shap](https://pypi.org/project/shap/0.54.1/) | Explaining model predictions with SHAP | MIT |
| [seaborn](https://pypi.org/project/seaborn/0.13.2/) | Charts for exploring data | BSD-3-Clause; includes other licenses too |
| [optuna-integration](https://pypi.org/project/optuna-integration/5.0.0/) | Helping tune models and stop weak trials early | MIT |
| [mljar-scikit-plot](https://pypi.org/project/mljar-scikit-plot/0.3.12/) | Charts for checking models; based on scikit-plot | MIT |
| [markdown](https://pypi.org/project/markdown/3.11.2/) | Markdown-to-HTML reports | BSD-3-Clause |
| [typing-extensions](https://pypi.org/project/typing-extensions/4.16.0/) | Supporting Python type hints | PSF-2.0 |
| [ipython](https://pypi.org/project/ipython/9.17.1/) | Displaying results in notebooks | BSD-3-Clause |
| [threadpoolctl](https://pypi.org/project/threadpoolctl/3.7.0/) | Controlling how many CPU threads libraries use | BSD-3-Clause |

### A few license details

- **Features Goldmine:** its [source code](https://github.com/mljar/features_goldmine/blob/main/LICENSE)
  uses Apache-2.0. The published 1.1.0 package is missing the license file and
  license information on PyPI. These should be added in a future release.
- **Matplotlib:** it has [its own license](https://matplotlib.org/stable/project/license.html),
  based on the Python Software Foundation license.
- **Packages that include other code:** keep the license files that come with them
  if you share a copy of the software.
- **mljar-scikit-plot:** thank you also to Reiichiro Nakano and the original
  scikit-plot project, which this package builds on.

## Optional packages

You only need these packages for certain features. RealMLP and TabM use pytabkit
and PyTorch. Kumo and TabICLv2 use structured-data-models, PyTorch, Hugging Face Hub,
and psutil. Mercury turns a trained model into a web app.

The `[all]` installation option adds the optional models. To add web apps too,
use `[all,apps]`. See the [installation guide](install.md) for commands.

| Package | Used for | License summary |
| --- | --- | --- |
| [mercury](https://pypi.org/project/mercury/3.2.12/) | Running prediction web apps | Apache-2.0 |
| [structured-data-models](https://pypi.org/project/structured-data-models/0.2.1/) | Running Kumo and TabICLv2 | Apache-2.0; includes BSD-3-Clause and other notices |
| [torch](https://pypi.org/project/torch/2.15.0/) | Running neural models and calculations | BSD-style license for core code; includes other licenses |
| [huggingface-hub](https://pypi.org/project/huggingface-hub/2.3.0/) | Downloading saved models | Apache-2.0 |
| [psutil](https://pypi.org/project/psutil/7.2.2/) | Resource and memory monitoring | BSD-3-Clause |
| [pytabkit](https://pypi.org/project/pytabkit/1.7.2/) | Running RealMLP and TabM mini | Apache-2.0 |

## Other packages we use

These packages are installed through the libraries above. MLJAR also uses them
directly. The libraries themselves may use more packages.

| Package | Used for | License summary |
| --- | --- | --- |
| [optuna](https://pypi.org/project/optuna/5.0.0/) | Finding good model settings | MIT |
| [packaging](https://pypi.org/project/packaging/26.3/) | Checking package versions and requirements | Apache-2.0 OR BSD-2-Clause |
| [matplotlib-inline](https://pypi.org/project/matplotlib-inline/0.2.2/) | Showing plots in notebooks | BSD-3-Clause |
| [pytorch-lightning](https://pypi.org/project/pytorch-lightning/2.6.6/) | Helping train models through pytabkit | Apache-2.0 |

## Models downloaded for you

Kumo and TabICLv2 download models that have already been trained. These saved
models have their own licenses, which can differ from the code used to run them.
The links below point to the exact versions used by this AutoML release.

| Model | Download source | Saved model license |
| --- | --- | --- |
| Kumo Tabular, small / medium / large, classifier and regressor | [Kumo Tabular — reviewed version](https://huggingface.co/nvidia/Kumo-Tabular/tree/3c3e10bbdb590ace29e7026847f92db3c603096d) | [OpenMDW License Agreement 1.1](https://huggingface.co/nvidia/Kumo-Tabular/blob/3c3e10bbdb590ace29e7026847f92db3c603096d/LICENSE) |
| TabICLv2 classifier and regressor, `v2-20260212` | [TabICLv2 — reviewed version](https://huggingface.co/jingang/TabICL/tree/4dcd344ece2c00be9e831fdd35bed57b5ad83e19) | BSD-3-Clause, listed in the [model description](https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/README.md) |

MLJAR saves the model descriptions and license files with these models. Keep
those files if you share a saved model or a prediction app.

RealMLP and TabM mini learn from your data without downloading a pretrained model.

## Code we have adapted

Our TabM training code builds on pytabkit. Thank you to David Holzmüller and
contributors. We keep their credit in the code and include their Apache-2.0
license in [`LICENSE-pytabkit.txt`](https://github.com/mljar/mljar-supervised/blob/master/LICENSE-pytabkit.txt).

## Tools that help us build AutoML

We use these tools to test the code and create this documentation. You do not need
to install them just to train a model. We also use psutil, listed above, during development.

| Package | Used for | License summary |
| --- | --- | --- |
| [pytest](https://pypi.org/project/pytest/9.2.0/) | Tests | MIT |
| [black](https://pypi.org/project/black/26.10.1/) | Formatting | MIT |
| [pytest-cov](https://pypi.org/project/pytest-cov/7.1.0/) | Checking which code is tested | MIT |
| [coveralls](https://pypi.org/project/coveralls/4.1.0/) | Sharing test coverage reports | MIT |
| [mkdocs-material](https://pypi.org/project/mkdocs-material/9.8.0/) | Building the docs website and its images | MIT |
| [Pygments](https://pypi.org/project/Pygments/2.21.0/) | Coloring code examples | BSD-2-Clause |
| [mkdocstrings](https://pypi.org/project/mkdocstrings/1.0.8/) | API documentation | ISC |
| [mkdocstrings-python](https://pypi.org/project/mkdocstrings-python/2.0.10/) | Generating Python API documentation | ISC |

## About this list

Last checked: **29 October 2026**, for **MLJAR AutoML 1.4.0b2**.

This page lists the packages named in our installation files, the other packages
our code uses directly, and our development tools. It does not list every package
used inside those projects. Your installed versions may also differ from the
versions we checked.

Python, system libraries, fonts, and tutorial datasets can have their own licenses
too. Check the files included with the software or data you share.

[Download the detailed package list](assets/acknowledgments/dependency-license-inventory.json)
for the versions and sources we checked. This JSON file is a review record,
not a list of everything installed on your computer.

## Giving credit when you share your work

- **Link to the projects you use.** This helps others discover their work.
- **Cite their research.** Each project explains how its authors would like to be cited.
- **Keep license files and copyright notices.** A thank-you page does not replace them.
- **List what you ship.** If you distribute an app or software package, record the
  exact versions and licenses included with it, including any downloaded models.

For a software release, that detailed list is often called a **software bill of
materials**, or **SBOM**. Think of it as an ingredients list for software.
It needs to be made from the actual release because different computers and
installation options can use different packages. This page is not a full SBOM.

There are useful standards for keeping this information clear:

| Standard | What it helps with |
| --- | --- |
| [SPDX](https://spdx.org/licenses/) | Gives licenses consistent names, such as `MIT` and `Apache-2.0`, and provides a format for software inventories. |
| [CycloneDX](https://cyclonedx.org/capabilities/sbom/) | Stores a software ingredients list in a format that tools can read. |
| [REUSE](https://reuse.software/spec-3.3/) | Records who owns the copyright and which license applies to each source file. |
| [Citation File Format](https://github.com/citation-file-format/citation-file-format) | Tells researchers how to cite a project, using a `CITATION.cff` file. |

For MLJAR releases, we recommend sharing this credits page alongside the license
notices and a release-specific SBOM. Tools such as
[CycloneDX Python](https://cyclonedx-bom-tool.readthedocs.io/en/latest/usage.html)
can help collect the installed Python packages. Downloaded models and system
libraries need to be checked too.
