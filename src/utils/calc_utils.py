import re
import string


def preprocess_text(text: str) -> str:
    """
    Preprocesses text for ASR evaluation.
    Converts to lowercase, removes punctuation, and normalizes whitespace.
    """
    if not text:
        return ""

    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Normalize whitespace (replace multiple spaces with a single space)
    text = " ".join(text.split())

    return text
