# AI-Powered Customer Review Sentiment Analysis

An end-to-end PyTorch project that converts Yelp star ratings into three sentiment classes:
**Negative** (1-2 stars), **Neutral** (3 stars), and **Positive** (4-5 stars). It implements
and compares a bidirectional LSTM and fine-tuned DistilBERT.

## Project structure

```text
app.py                  Streamlit entry point
train.py                Unified training entry point
evaluation.py           Test-set evaluation and charts
src/data/               Dataset download, cleaning, sampling, and EDA
src/preprocessing/      Vocabulary and sequence/tokenization utilities
src/models/             BiLSTM and DistilBERT implementations
src/evaluation/         Evaluation package
data/raw/               Downloaded Yelp splits
data/processed/         Train, validation, and test CSV files
models/                 Saved model artifacts
outputs/                Metrics, histories, and charts
reports/                Project reports
notebooks/              Optional analysis notebooks
```

## Setup

The project uses Python 3.11. Run commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The environment automatically uses CUDA when available and CPU otherwise.

## Data preparation

The data pipeline uses `Yelp/yelp_review_full` (the canonical public copy; the original
`fancyzhx/yelp_review_full` endpoint may require authentication), removes missing/duplicate text, prevents
cross-split text leakage, and creates a stratified validation set from training data.

CPU-oriented defaults are 5,000 training examples and 1,250 test examples. The requested
20,000/5,000 scale can be used with the override below:

```powershell
python -m src.data.prepare_data --train-size 5000 --test-size 1250 --seed 42
# Larger personal-computer run:
python -m src.data.prepare_data --train-size 20000 --test-size 5000 --seed 42
```

Outputs:

- `data/raw/yelp_train.csv`
- `data/raw/yelp_test.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `outputs/eda_overview.png`
- `outputs/sentiment_by_split.png`
- `outputs/top_words.png`
- `outputs/data_summary.json`

## Train models

Prepare and train both models:

```powershell
python train.py --prepare-data --model all --epochs 1 --batch-size 8 --seed 42
```

Train only one model:

```powershell
python train.py --model bilstm --epochs 3 --batch-size 32 --max-length 256
python train.py --model distilbert --epochs 2 --batch-size 8 --max-length 128
```

For a larger data run:

```powershell
python train.py --prepare-data --train-size 20000 --test-size 5000 --model all --epochs 2 --batch-size 8
```

Saved artifacts:

- `models/bilstm.pt`
- `models/bilstm_vocab.json`
- `models/distilbert/`
- `outputs/*_history.json`
- `outputs/training_times.json`

## Evaluate

Run evaluation after training:

```powershell
python evaluation.py --model all
python evaluation.py --model bilstm
python evaluation.py --model distilbert
```

Actual test predictions generate `outputs/evaluation_metrics.json`, confusion matrices,
and `outputs/model_comparison.png`. No performance values are written until predictions exist.

## Run Streamlit

```powershell
streamlit run app.py
```

Open the local URL displayed by Streamlit. The app includes Home, Single Review Prediction,
Bulk Review Analysis, Analytics Dashboard, and Model Comparison pages.

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` and reinstall `requirements.txt`.
- Dataset or model download failure: verify internet access and Hugging Face availability.
- Out of memory: reduce `--batch-size`, `--max-length`, and data sample sizes.
- Slow CPU training: reduce epochs and use 1,000-5,000 training rows for initial tests.
- CUDA unavailable: this is normal on CPU-only machines; training remains supported on CPU.

## Resume-ready project description

Built an end-to-end customer sentiment classification system on Yelp reviews using a PyTorch
BiLSTM and fine-tuned DistilBERT. Implemented reproducible stratified sampling, leakage-safe
train/validation/test splits, training-only vocabulary construction, weighted loss, real metric
evaluation, confusion matrices, inference benchmarking, and a five-page Streamlit application.
