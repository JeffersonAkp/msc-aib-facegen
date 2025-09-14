from datasets import load_dataset

def get_fairface(split="train", subset="0.25"):
    """
    subset: '0.25' (léger) ou '1.0' (complet si dispo)
    """
    ds = load_dataset("HuggingFaceM4/FairFace", subset)
    return ds[split]
