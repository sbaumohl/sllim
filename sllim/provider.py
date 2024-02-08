import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Provider:
    def call():
        pass
    
    def chat():
        pass
    
    def complete():
        pass
    
    def embed():
        pass

# https://platform.openai.com/docs/api-reference
class SlimOpenAI(Provider):
    pass

# https://docs.cohere.com/reference
class SlimCohere(Provider):
    pass

# https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources
class SllimClaude(Provider):
    pass