# News Topic Classifier (AG News)

Fine-tunes a pretrained Transformer to classify news articles into **4 topics** (World, Sports, Business, Sci/Tech) using the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) dataset.

## What it does
- Loads AG News with 🤗 `datasets` and tokenizes with `AutoTokenizer` + dynamic padding (`DataCollatorWithPadding`).
- Fine-tunes `AutoModelForSequenceClassification` (4 labels) two ways:
  - the high-level **`Trainer`** API
  - a **manual PyTorch training loop** (AdamW + linear warmup scheduler)
- Evaluates with the `evaluate` library (the notebook reports an F1 of about **0.94**).

## Stack
`PyTorch` · `Hugging Face Transformers` · `Datasets` · `Evaluate`

## Run it
Open `News_Topic_Classifier.ipynb` in Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU).
