def split_text(nlp_pipeline, text):
    """Split the text in sentences and tokenize each sentence.

    >>> import stanza
    >>> stanza.download(lang="en", processors="tokenize", verbose=False)
    >>> nlp = stanza.Pipeline(lang="en", processors="tokenize", verbose=False)
    >>> split_text(nlp, "First sentence. Second sentence.")
    [['First', 'sentence', '.'], ['Second', 'sentence', '.']]
    """
    doc = nlp_pipeline(text)
    return [
        [token.text for token in sentence.tokens] for sentence in doc.sentences
    ]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
