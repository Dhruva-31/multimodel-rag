import re

# Matches: (resume.pdf, p.3) or (notes.txt, ¶12) or (scan.png)
_CITATION_RE = re.compile(
    r"\((?P<filename>[^\s,)]+\.[a-zA-Z0-9]+)"
    r"(?:,\s*(?:p\.|¶)\s*(?P<locator>\d+))?"
    r"\)",
    re.IGNORECASE,
)


def _build_index(
    reranked: list[tuple[str, float, dict]],
) -> set[tuple[str, str | None]]:
    """
    Build a set of (filename, locator) pairs from retrieved chunks.
    locator is page number or paragraph number as a string, or None for images.
    """
    index = set()

    for _, _, metadata in reranked:
        filename = metadata.get("filename", "").lower()
        page = metadata.get("page")
        paragraph = metadata.get("paragraph")

        if page is not None:
            index.add((filename, str(page)))

        if paragraph is not None:
            index.add((filename, str(paragraph)))

        if page is None and paragraph is None:
            index.add((filename, None))

    return index


def check(response: str, reranked: list[tuple[str, float, dict]]) -> str:
    """
    Verify every citation in response against retrieved chunks.
    Appends [?] to any citation that cannot be verified.
    Returns the cleaned response.
    """
    index = _build_index(reranked)

    result = response
    offset = 0

    for match in _CITATION_RE.finditer(response):

        filename = match.group("filename").lower()
        locator = match.group("locator")

        verified = (filename, locator) in index

        if not verified:

            flag = "[?]"

            insert_pos = match.end() + offset

            result = result[:insert_pos] + flag + result[insert_pos:]

            offset += len(flag)

    return result
