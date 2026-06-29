import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class Generator:

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
    ):
        self.model = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(
        self,
        prompt: str,
    ) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content or ""
