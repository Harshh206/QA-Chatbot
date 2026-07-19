from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Optional, List, Dict, Any
import logging
from config import config
from prompts.system_prompt import system_prompt as default_system_prompt

logger = logging.getLogger(__name__)

class LLMManager:
    """Simple LLM manager for Ollama"""

    def __init__(
        self,
        model_name: str = config.llm,
        base_url: str = config.base_url,
        temperature: float = config.temperature,
        system_prompt: str = default_system_prompt,
        **kwargs,
    ):
        self.system_prompt = system_prompt

        # Initialize the LLM
        self.llm = ChatOllama(
            model=model_name, 
            base_url=base_url, 
            temperature=temperature, 
            **kwargs
        )
        logger.info(f"Initialized LLM: {model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override

        Returns:
            Generated response
        """

        active_system_prompt = (
            self.system_prompt if system_prompt is None else system_prompt
        )
        messages = []
        if active_system_prompt:
            messages.append(SystemMessage(content=active_system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            # Handle both string and list responses from ChatOllama
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                # Join list content into a single string
                return "".join(str(item) for item in response.content)
            else:
                return str(response.content)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

    def generate_with_context(self, query: str, context: str) -> str:
        """
        Generate response with context (for RAG)

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Generated response
        """
        prompt = f"""Context: {context}

Question: {query}

Answer based on the context above:"""

        return self.generate(prompt)

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Stream generate a response
        """
        active_system_prompt = (
            self.system_prompt if system_prompt is None else system_prompt
        )
        messages = []
        if active_system_prompt:
            messages.append(SystemMessage(content=active_system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            for chunk in self.llm.stream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming: {e}")
            yield f"Error: {str(e)}"

    def validate(self, system_prompt: Optional[str] = None) -> bool:
        """Test if LLM is working"""
        try:
            response = self.generate("Hello, respond with 'OK'", system_prompt)
            return "OK" in response or len(response) > 0
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return False
