import logging

import cohere
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# # https://docs.cohere.com/reference
# class SllimCohere(Provider):
#     def __init__(self, api_key: str) -> None:
#         self.client = cohere.Client(api_key)
#         super().__init__()


# # https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources
# class SllimClaude(Provider):
#     def __init__(self) -> None:
#         super().__init__()
#         raise NotImplementedError