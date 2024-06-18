# DiamondPricePrediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Diamond price prediction based on given features.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for src
│                         and configuration for tools like black
|
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src                <- Source code for use in this project.
|   │
|   ├── __init__.py    <- Makes src a Python module
|   │
|   ├── data           <- Scripts to download or generate data
|   │   └── data_ingestion.py
|   |   └── data_transformation.py
|   │
|   ├── models         <- Scripts to train models and then use trained models to make
|   |   │                 predictions
|   |   ├── model_trainer.py
|   |   └── model_evaluation.py
|   |
|   ├── logger.py      <- contains class for logging
|
└── templates
    └── index.html
|
└── app.py              <- flask app for deployment    
```

--------

