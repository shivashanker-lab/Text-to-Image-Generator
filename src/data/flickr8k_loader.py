from datasets import load_dataset


DATASET_ID = "intro/flickr8k"


def load_flickr8k(split="train", streaming=True):
    """
    Load the Flickr8k image-caption dataset.

    Parameters
    ----------
    split : str, default="train"
        Dataset split to load: "train", "dev", or "test".
    streaming : bool, default=True
        Whether to load the dataset in streaming mode.

    Returns
    -------
    Dataset or IterableDataset
        The requested Flickr8k split.
    """
    return load_dataset(
        DATASET_ID,
        split=split,
        streaming=streaming,
    )
