# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run and split_text_on_tokens methods definition."""
import re
from collections.abc import Iterable
from typing import Any, List, Optional

import tiktoken
from datashaper import ProgressTicker
from langchain.text_splitter import RecursiveCharacterTextSplitter

import graphrag.config.defaults as defs
from graphrag.index.text_splitting import Tokenizer
from graphrag.index.verbs.text.chunk.typing import TextChunk


def run(
    input: list[str], args: dict[str, Any], tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    tokens_per_chunk = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)


    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
        chunk_overlap=chunk_overlap,  #### update
        tokens_per_chunk=tokens_per_chunk  ### update
    )


# Adapted from - https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471
# So we could have better control over the chunking process
def split_text_on_tokens(
        texts: list[str], enc: Tokenizer, tick: ProgressTicker, chunk_overlap,
        tokens_per_chunk  # update
) -> list[TextChunk]:
    """Split incoming text and return chunks."""
    result = []
    mapped_ids = []

    # for source_doc_idx, text in enumerate(texts):
    #     encoded = enc.encode(text)
    #     tick(1)
    #     mapped_ids.append((source_doc_idx, encoded))

    # input_ids: list[tuple[int, int]] = [
    #     (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    # ]
    for source_doc_idx, text in enumerate(texts):
        tick(1)
        mapped_ids.append((source_doc_idx, text))

    # added by congbo
    def length_function(text: str) -> int:
        return len(enc.encode(text))

    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True, is_separator_regex=True, chunk_size=tokens_per_chunk,
        chunk_overlap=chunk_overlap, length_function=length_function
    )

    for source_doc_idx, text in mapped_ids:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            result.append(
                TextChunk(
                    text_chunk=chunk,
                    source_doc_indices=[source_doc_idx] * len(chunk),
                    n_tokens=len(chunk),
                )
            )
    # start_idx = 0
    # cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    # chunk_ids = input_ids[start_idx:cur_idx]
    # while start_idx < len(input_ids):
    #     chunk_text = enc.decode([id for _, id in chunk_ids])
    #     doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
    #     result.append(
    #         TextChunk(
    #             text_chunk=chunk_text,
    #             source_doc_indices=doc_indices,
    #             n_tokens=len(chunk_ids),
    #         )
    #     )
    #     start_idx += enc.tokens_per_chunk - enc.chunk_overlap
    #     cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    #     chunk_ids = input_ids[start_idx:cur_idx]

    return result


# -----------------------------------------------------------------------------------
# 适用中文
def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]