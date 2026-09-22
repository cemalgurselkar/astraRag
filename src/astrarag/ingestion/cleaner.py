import re
import unicodedata


class TextCleaner:
    """Normalize common text extraction artifacts before chunking."""

    _SOFT_HYPHEN = "\u00ad"
    _REPLACEMENT_CHARACTER = "\ufffd"
    _NONCHARACTER = "\ufffe"

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)

        # Characters commonly introduced by PDF text extraction.
        text = text.replace(self._SOFT_HYPHEN, "")
        text = text.replace(self._REPLACEMENT_CHARACTER, "")
        text = text.replace(self._NONCHARACTER, "")

        # Join words explicitly hyphenated across line boundaries:
        # "lan-\nguage" -> "language"
        text = re.sub(
            r"(?<=\w)-\s*\n\s*(?=\w)",
            "",
            text,
        )

        # PDF line layout is not useful for our retrieval representation.
        text = re.sub(r"\s*\n\s*", " ", text)

        # Normalize repeated whitespace.
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()