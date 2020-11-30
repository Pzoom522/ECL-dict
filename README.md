# ECL-map
Dictionary builder for Endangered Chinese Languages :book:

-------

## Crawl data for endangered Chinese languages
- We've already collected urls of target languages (see `language_url_dict.json`)
- `spider_0.py` will store the href lists of all xml files of each language and store them under `result`
- `spider_1.py` will fetch all xml resources (i.e., transcription of recordings)

## OCR
- Code is copied from [Paddle's lightweight OCR tool](https://github.com/PaddlePaddle/PaddleOCR)
- We execute the program to get text from *Story China (aka 故事会)*. Due to request from our copyright provider, the full dataset is not publicly available at present. We only provide a sample and plan to release the complete corpus soon.

## Word embedding based on PMI+SVD
- Code is copied from [svd2vec](https://github.com/valentinp72/svd2vec)
- It has promising performance when training with small corpus

## Word embedding mapping
- We provide four modes: supervised, semi-supervised, identical and unsupervised. For endangered languages, we recommend semi-supervised and unsupervised modes.
- Usage: `python map_embeddings.py --MODE [path/to/small/dict] path/to/src/emb path/to/tgt/emb path/to/mapped/src/emb path/to/mapped/tgt/emb`
- Dictionaries can be directly retrived from the two aligned embeddings
