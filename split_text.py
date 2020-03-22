import stanza


def split_text(nlp_pipeline, text):
    "Split the text in sentences and tokenize each sentence"
    doc = nlp_pipeline(text)
    return [
        [token.text for token in sentence.tokens] for sentence in doc.sentences
    ]


### Unit test ###

nlp = stanza.Pipeline(
    lang="en", processors="tokenize", use_gpu=False, verbose=False
)
t = split_text(nlp, "This is a sentence. This is another sentence.")
assert t[0] == ["This", "is", "a", "sentence", "."]
assert t[1] == ["This", "is", "another", "sentence", "."]
