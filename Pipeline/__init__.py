import time
from typing import Optional, Dict, Any, List, Union
from langchain_together import ChatTogether
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import os
from dotenv import load_dotenv

load_dotenv()



class LLMMetadata(BaseModel):
    service_provider: str = Field(description="The service provider used (e.g., 'openai', 'groq', 'huggingface')")
    llm_model_name: str = Field(description="The name of the model used (e.g., 'gpt-4', 'llama-3.1')")
    temperature: float = Field(description="Sampling temperature used for generation")
    max_tokens: int = Field(description="Maximum number of tokens in the response")
    response_time_seconds: float = Field(description="Time taken to generate the response")
    error: Optional[str] = Field(default=None, description="Error message if the invocation failed")


class LLMResponse(BaseModel):
    content: str = Field(description="Response content from the LLM")
    metadata: LLMMetadata = Field(description="Metadata associated with the response")


class BaseLLM(BaseChatModel):
    service_provider: str = Field(description="Service provider (e.g., 'openai', 'groq', 'huggingface')", default=os.environ.get('service_provider'))
    llm_model_name: str = Field(description="Name of the model (e.g., 'gpt-4', 'llama-3.1')", default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=1024, description="Maximum number of tokens in the response")
    # api_key: str = Field(description='API Key of the Service Provider')
    base_url: str = Field(description='Optional Base URL for Open AI Service Provider', default=None)

    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)
    _conversation_history: List[Dict[str, str]] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_llm()

    @property
    def _llm_type(self) -> str:
        return f"{self.service_provider}_{self.llm_model_name}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "service_provider": self.service_provider,
            "llm_model_name": self.llm_model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def initialize_llm(self):
        if self.service_provider.lower() == 'groq':
            self._llm = ChatGroq(
                model=os.environ.get('llm_model_name'),
                temperature=self.temperature,
                api_key=os.environ.get('api_key')
            )
        elif self.service_provider.lower() == "openai":
            self._llm = OpenAI(
                api_key=os.environ.get('api_key'),
                base_url=os.environ.get('base_url')
            )
        elif self.service_provider.lower() == "togetherai":
            self._llm = ChatTogether(
                model=os.environ.get('llm_model_name'),
                api_key=os.environ.get('api_key')
            )


        else:
            raise ValueError(f"Unsupported service provider: {self.service_provider}")

    def _generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        if not self._llm:
            raise ValueError("LLM not initialized properly.")

        if self.service_provider.lower() in ["togetherai", "groq"]:
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

        elif self.service_provider.lower() == "openai":
            chat_completion = self._llm.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            content = chat_completion.choices[0].message.content
        else:
            raise ValueError(f"Unsupported service provider: {self.service_provider}")

        return content

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        start_time = time.time()
        try:

            content = self._generate(prompt)

            end_time = time.time()
            response_time = end_time - start_time
            metadata = LLMMetadata(
                service_provider=self.service_provider,
                llm_model_name=self.llm_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_time_seconds=response_time,
                error=None
            )
            return LLMResponse(content=content, metadata=metadata)
        except Exception as e:
            print(e)
            end_time = time.time()
            response_time = end_time - start_time
            metadata = LLMMetadata(
                service_provider=self.service_provider,
                llm_model_name=self.llm_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_time_seconds=response_time,
                error=str(e)
            )
            return LLMResponse(content="", metadata=metadata)