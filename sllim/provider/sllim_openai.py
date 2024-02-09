from typing import Optional, TypeVar, TypedDict, Callable

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types import Completion

import tiktoken

from ..provider import Provider, logger, FunctionT, Prompt
from .. import cache, catch

# https://platform.openai.com/docs/api-reference
class SllimOpenAI(Provider):
    def __init__(self, api_key: str | None = None) -> None:
        """Instantiate a Sllim OpenAI client wrapper.

        Args:
            api_key (str | None, optional): OpenAI API Key. If none is provided, OpenAI attempts
                to retrieve it from environment variables. Defaults to None.
        """
        self.client = OpenAI(api_key=api_key)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        super().__init__()
        
    def set_api_key(self, api_key=None):
        self.client.api_key = api_key
    
    def get_token_counts(self):
        return self.prompt_tokens, self.completion_tokens

    def reset_token_counts(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    def estimate(self, messages_or_prompt: Prompt, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        total = 0

        if isinstance(messages_or_prompt, str):
            total = len(enc.encode(messages_or_prompt))
        else:
            total = sum([
                len(enc.encode(text["content"])) 
                for text in messages_or_prompt
            ])

        model_cost = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "text-davinci-003": 0.02,
            "text-davinci-002": 0.012,
        }
        return {"tokens": total, "cost": total * model_cost.get(model) / 1000}
    
    @catch
    @cache
    def chat(
        self,
        messages,
        model="gpt-3.5-turbo",
        max_tokens: Optional[int] = None,
        functions: Optional[list[FunctionT]] = None,
        temperature: float = 1,
        top_p: float = 1,
        n: int = 1,
        stop: Optional[str | list[str]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Optional[dict[int, float]] = None,
        deployment_id: Optional[str] = None,
        cache_version: Optional[str] = None,
    ) -> str:

        if deployment_id:
            model = None

        default_params = {
            "model": None,
            "max_tokens": None,
            "temperature": 1,
            "top_p": 1,
            "n": 1,
            "stop": None,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "logit_bias": None,
            "deployment_id": None,
        }
        kwargs = {
            k: v
            for k, v in locals().items()
            if k in default_params and v != default_params[k]
        }
        max_tokens_str = "infinity" if max_tokens is None else str(max_tokens)
        model_str = model if model else deployment_id
        logger.info(
            f"Calling {model_str} using at most {max_tokens_str} with messages: {messages}"
        )

        response: ChatCompletion = self.client.chat.completions.create(
            messages=messages,
            **kwargs,
        )

        message = response.choices[0].message
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens

        if message.content:
            logger.info(f"Response: {message.content}")
            return message.content

        raise Exception("No content found in response.")


    @catch
    @cache
    def call(
        self,
        messages,
        model="gpt-3.5-turbo",
        max_tokens: int = 256,
        functions: Optional[list[FunctionT]] = None,
        temperature: float = 1,
        top_p: float = 1,
        n: int = 1,
        stop: Optional[str | list[str]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Optional[dict[int, float]] = None,
        cache_version: Optional[str] = None,
    ) -> str:
        """Generate a function call based on the messages. Use `create_function_call` to prepare `functions` arg."""

        default_params = {
            "temperature": 1,
            "top_p": 1,
            "n": 1,
            "functions": None,
            "stop": None,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "logit_bias": None,
        }
        kwargs = {
            k: v
            for k, v in locals().items()
            if k in default_params and v != default_params[k]
        }

        response: ChatCompletion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        message = response.choices[0].message
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens

        if message.function_call:
            return message.function_call

        raise Exception("No function call found in response. %s" % str(message))


    @catch
    @cache
    def complete(
        self,
        prompt: str = "<|endoftext|>",
        model: str = "text-davinci-003",
        max_tokens: int = 16,
        temperature: float = 1,
        top_p: float = 1,
        n: int = 1,
        logprobs: Optional[int] = None,
        stop: Optional[str | list[str]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        best_of: int = 1,
        logit_bias: Optional[dict[int, float]] = None,
        cache_version: Optional[str] = None,
    ) -> str:
        global prompt_tokens, completion_tokens

        default_params = {
            "prompt": "<|endoftext|>",
            "temperature": 1,
            "top_p": 1,
            "n": 1,
            "max_tokens": 16,
            "logprobs": None,
            "stop": None,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "best_of": 1,
            "logit_bias": None,
        }
        kwargs = {
            k: v
            for k, v in locals().items()
            if k in default_params and v != default_params[k]
        }

        response: Completion = self.client.completions.create(
            model=model,
            max_tokens=max_tokens,
            **kwargs,
        )
        message = response.choices[0].text
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens

        return message


    @catch
    @cache
    def embed(
        self,
        text: str | list[str], engine="text-embedding-ada-002", deployment_id: str = None
    ) -> list[float] | list[list[float]]:
        if deployment_id:
            engine = None
        kwargs = {
            k: v
            for k, v in locals().items()
            if k in ["engine", "deployment_id"] and v is not None
        }
        if isinstance(text, list):
            text = [t.replace("\n", " ") for t in text]
            response = self.client.embeddings.create(input=text, **kwargs)["data"]
            return [x["embedding"] for x in response]
        else:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], **kwargs)["data"][0]["embedding"]
