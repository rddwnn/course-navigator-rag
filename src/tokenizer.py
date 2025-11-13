import tiktoken
from functools import lru_cache


@lru_cache(maxsize=4)
def get_tokenizer(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = get_tokenizer(model)
    return len(enc.encode(text))


def split_into_token_chunks(
    text: str,
    max_tokens: int,
    model: str,
):

    enc = get_tokenizer(model)
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks
