# IEEEBigData2019-TablesEmbeddings

This repository contains resources developed within the following paper:
M. Trabelsi, B. D. Davison and J. Heflin, "Improved Table Retrieval Using Multiple Context Embeddings for Attributes," 2019 IEEE International Conference on Big Data (Big Data), Los Angeles, CA, USA, 2019, pp. 1238-1244.

## Requirements

- Python 3.6
- Tensorflow-gpu 2.0.0a0
- elasticsearch 5.5.3
- Nordlys (https://github.com/iai-group/nordlys)
- fasttext embedding (https://fasttext.cc/docs/en/english-vectors.html)
- trec_eval is used to calculate evaluation metrics (https://github.com/usnistgov/trec_eval)

## Data

[`WikiTables` corpus](http://iai.group/downloads/smart_table/WP_tables.zip) contains over 1.6𝑀 tables that are extracted from Wikipedia. Each table has five indexable fields: table caption, attributes (column headings), data rows, page title, and section title. Download and uncrompress the `WikiTables` corpus. We use the same queries that were used by [Zhang and Balog](https://github.com/iai-group/www2018-table), where every query-table pair is evaluated using three numbers: 0 means “irrelevant”, 1 means “partially relevant” and 2 means “relevant”.

## Word embeddings for attribute tokens

The first step is to prepare the target-context pairs for the `WikiTables` corpus:
```bash
python prepare_target_context_data.py
```
Please set the `data_folder` variable in `prepare_target_context_data.py` to the corresponding folder of `WikiTables`.

Then, the second step is to train the embedding using the adapted Skipgram model:

```bash
python train_embeddings.py
```

## Table features

The next step is to extract the original and augmented table features that should be indexed for table retrieval:

```bash
python prepare_elasticsearch_input.py
```

Then these features are indexed for `elasticsearch`:

```bash
python elasticsearch_index_data.py
```
Please set the `'/path/to/nordlys'` in the code to point to `nordlys` folder.

## Ranking scores

Compute the `MaxTable` ranking scores:

```bash
python prepare_ranking_input.py
```

Find the multi-field weights using grid search, and report the ranking results:

```bash
python grid_search_parameters.py
```


## Citation

@INPROCEEDINGS{9005681,
  author={M. {Trabelsi} and B. D. {Davison} and J. {Heflin}},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)}, 
  title={Improved Table Retrieval Using Multiple Context Embeddings for Attributes}, 
  year={2019},
  volume={},
  number={},
  pages={1238-1244},}
  
## Contact
  
  if you have any questions, please contact Mohamed Trabelsi at mot218@lehigh.edu
  
  
