# Toxic Comment Classification

## Final Report

## Final Code Submission

#### How to run this code

1. Clone the repository or open in a Codespace

```
git clone https://github.com/tmetz/toxic_comment_classification.git
```

2. Run `classify.py`.  This will unzip `glove.6B.50d.txt.zip` into a new directory, `glove`, and then run the entire pipeline.

```
python classify.py
```

#### Description of the files and folders

| File                          | Description                                                                                          |
|-------------------------------|------------------------------------------------------------------------------------------------------|
| `.gitignore`                  | Makes sure we don't check the `glove` directory into Git since it has a large file                   |
| `README.md`                   | This file                                                                                            |
| `Toxic Content Filtering.pdf` | PDF of Presentation                                                                                  |
| `classify.py`                 | Main codebase                                                                                        |
| `glove.6B.50d.txt.zip`        | Zipped GloVe pretrained word embeddings                                                              |
| `test.csv`                    | Test data                                                                                            |
| `test_labels.csv`             | Test labels                                                                                          |
| `train.csv`                   | Training data (not strictly necessary to have the CSV since we are able to load this via `datasets`) |

## Final Video Presentation

Visit https://www.youtube.com/watch?v=d3DB9UrTa8M to watch the presentation.

## Slide Deck

The slide deck can be found in this repo ([click here](https://github.com/tmetz/toxic_comment_classification/blob/main/Toxic%20Content%20Filtering.pdf))