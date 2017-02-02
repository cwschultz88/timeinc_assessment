# Time Inc Data Science Homework Assignment

This repo was created as part of Time Inc.'s application process.

Written for Mac OS - Code should work in a Linux environment but not for Windows environment because file paths were not written to be universal in the code.

Assumes all python scripts are called from the root of the repo, i.e. this directory level.

Structure of Repo Files:
```
  data/                 contains data sources for project
     - raw/                   contains raw unprocessed data
     - processed/             contains processed data that can used to train models and model outputs
     - misc/                  contains misc data
  excel_worksheets/     contains excel analysis/visualization files
  models/               contains pickled model files and training output
  src/                  contains python source files
      - data/                  contains data retriever iterator getter functions
      - features/              contains model feature building code, built features end up in data/processed/
      - models/                contains model training and analysis code, trained models are pickled and stored in models/
  tests/                contains unit tests for src code
      - data/                  contains unit tests for data retriever code
      - features/              contains unit tests for building features code
      - models/                contains unit tests for model code
  requirements.txt      pip install requirements for this repo
  README.md             this readme
```

To run unit tests, make sure to use -m unittest flag with the python interpreter. For example:
```
python -m unittest tests.data.raw.twitter_tests
```
