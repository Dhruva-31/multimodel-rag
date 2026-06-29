import re

_GREETINGS = re.compile(
    r"^\s*(hi|hello|hey|howdy|sup|yo|thanks?|thank you|bye|goodbye|ok|okay|sure|cool|great|nice)"
    r"(\s+(there|friend|bot|assistant|how are you|how r u|what'?s up|wassup|buddy|man))?"
    r"\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_SMALL_TALK = re.compile(
    r"^\s*(how are you|how r u|how are u|what'?s up|wassup|who are you|what are you"
    r"|are you (an? )?(ai|bot|assistant|llm)|what can you do)"
    r"\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_PROMPT_INJECTION = re.compile(
    r"(ignore.*instructions"
    r"|ignore.*context"
    r"|forget.*instructions"
    r"|forget.*context"
    r"|use.*outside knowledge"
    r"|answer.*from your own knowledge"
    r"|disregard.*previous)",
    re.IGNORECASE,
)


def check(query: str) -> tuple[bool, str]:

    query = query.strip()

    if not query:
        return (
            False,
            "Please enter a question about your document.",
        )

    if len(re.findall(r"[a-zA-Z]", query)) == 0:
        return (
            False,
            "That doesn't look like a question. Please ask something about your document.",
        )

    if _GREETINGS.fullmatch(query):
        return (
            False,
            "Hi! I'm here to answer questions about your uploaded document. What would you like to know?",
        )

    if _SMALL_TALK.fullmatch(query):
        return (
            False,
            "I'm a document assistant and can only answer questions about the uploaded document.",
        )

    if _PROMPT_INJECTION.search(query):
        return (
            False,
            "Please ask questions related to the uploaded document.",
        )

    if len(query) < 3:
        return (
            False,
            "Please provide a more specific question about your document.",
        )

    if len(query) > 2000:
        return (
            False,
            "Your question is too long. Please shorten it and try again.",
        )

    return (
        True,
        "",
    )
