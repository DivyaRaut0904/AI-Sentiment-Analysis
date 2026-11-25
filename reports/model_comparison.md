# Model Comparison Report

## Experimental Setup

- Dataset: Yelp Review Full
- Sentiment mapping: 1-2 stars = Negative, 3 stars = Neutral, 4-5 stars = Positive
- CPU-adjusted split: 800 training, 200 validation, 500 test rows
- Random seed: 42
- Training epochs: 1 for both models
- Validation split: stratified 20% of the sampled training data
- Framework: PyTorch

## Aggregate Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Training Time (s) | Test Inference (s) |
|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | 0.4100 | 0.3845 | 0.3717 | 0.3480 | 2.34 | 0.33 |
| DistilBERT | 0.6700 | 0.6393 | 0.6250 | 0.6277 | 265.37 | 37.61 |

These metrics were calculated from actual test predictions. The primary comparison metric
is macro F1 because the project uses three sentiment classes.

## Per-Class DistilBERT Results

| Sentiment | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.7550 | 0.7550 | 0.7550 | 200 |
| Neutral | 0.3008 | 0.4000 | 0.3433 | 100 |
| Positive | 0.8623 | 0.7200 | 0.7847 | 200 |

## Per-Class BiLSTM Results

| Sentiment | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Negative | 0.4269 | 0.7150 | 0.5346 | 200 |
| Neutral | 0.2432 | 0.1800 | 0.2069 | 100 |
| Positive | 0.4835 | 0.2200 | 0.3024 | 200 |

## Conclusion

In this CPU-adjusted experiment, DistilBERT achieved the stronger accuracy and macro F1
after one training epoch. BiLSTM was substantially faster to train and perform inference,
but had weaker neutral and positive recall. These results are a preliminary personal-computer
baseline and should be improved with a larger sample, more epochs, and hyperparameter tuning
on GPU hardware before production use.
