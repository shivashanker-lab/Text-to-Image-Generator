import re

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 32


def clean_text(text):
    """
    Normalize a text description for preprocessing.

    Parameters
    ----------
    text : str
        Input text description.

    Returns
    -------
    str
        Lowercase text with punctuation removed and whitespace normalized.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_text_encoder(model_name=MODEL_NAME):
    """
    Load a tokenizer and BERT text encoder.

    Returns
    -------
    tuple
        (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def generate_text_embedding(
    text,
    tokenizer,
    model,
    max_length=MAX_LENGTH,
):
    """
    Generate a pooled BERT embedding for a text description.

    The tokenized inputs are automatically moved to the same
    device as the text encoder, allowing both CPU and CUDA use.

    Parameters
    ----------
    text : str
        Input text description.

    tokenizer : transformers tokenizer
        Tokenizer associated with the text encoder.

    model : transformers model
        BERT text encoder.

    max_length : int, default=32
        Maximum token sequence length.

    Returns
    -------
    torch.Tensor
        Mean-pooled embedding with shape [1, hidden_size].
    """
    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = (
        outputs.last_hidden_state * attention_mask
    )

    summed_embeddings = masked_embeddings.sum(dim=1)

    token_counts = (
        attention_mask.sum(dim=1).clamp(min=1)
    )

    pooled_embedding = (
        summed_embeddings / token_counts
    )

    return pooled_embedding
