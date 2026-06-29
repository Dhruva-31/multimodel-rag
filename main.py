from guardrails.output_guardrail import check as output_check

# Simulate reranked chunks with known metadata
reranked = [
    (
        "Dhruva has experience in Python and ML.",
        0.9,
        {
            "filename": "resume.pdf",
            "type": "pdf",
            "page": 1,
            "paragraph": None,
        },
    ),
    (
        "He interned at a fintech company.",
        0.8,
        {
            "filename": "resume.pdf",
            "type": "pdf",
            "page": 2,
            "paragraph": None,
        },
    ),
]

# Test 1 — all citations valid, nothing should be flagged
response_clean = (
    "Dhruva has strong Python skills (resume.pdf, p.1) "
    "and interned at a fintech company (resume.pdf, p.2)."
)

# Test 2 — p.99 was never retrieved, should get [?]
response_fake = (
    "Dhruva has strong Python skills (resume.pdf, p.1) "
    "and has a PhD from MIT (resume.pdf, p.99)."
)

# Test 3 — completely fabricated file, should get [?]
response_hallucinated = "Dhruva won a Nobel Prize (fake_document.pdf, p.5)."

print("Test 1 — clean citations:")
print(output_check(response_clean, reranked))
print()

print("Test 2 — fake page number:")
print(output_check(response_fake, reranked))
print()

print("Test 3 — hallucinated file:")
print(output_check(response_hallucinated, reranked))
