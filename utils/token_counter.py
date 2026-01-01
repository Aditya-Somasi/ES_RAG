import tiktoken
from typing import List, Dict, Any


class TokenCounter:
    """Token counter for Azure OpenAI GPT models."""
    
    # Azure OpenAI pricing (example rates, adjust as needed)
    # Prices per 1K tokens
    PRICING = {
        "gpt-4": {
            "input": 0.03,
            "output": 0.06,
        },
        "gpt-4-turbo": {
            "input": 0.01,
            "output": 0.03,
        },
        "gpt-35-turbo": {
            "input": 0.0015,
            "output": 0.002,
        },
        "gpt-4.1": {  # Pricing for GPT-4.1
            "input": 0.035,  # Example rate, adjust as needed
            "output": 0.07,  # Example rate, adjust as needed
        },
    }
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for token encoding
        """
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for GPT-4 family
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Total number of tokens
        """
        tokens = 0
        for message in messages:
            # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens += 4
            for key, value in message.items():
                if isinstance(value, str):
                    tokens += self.count_tokens(value)
                if key == "name":
                    tokens -= 1
        tokens += 2  # Every reply is primed with <im_start>
        return tokens
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str = "gpt-4"
    ) -> float:
        """
        Estimate cost for Azure OpenAI API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Model name for pricing lookup
            
        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        if "gpt-4.1" in model_name.lower():  # Handle GPT-4.1 explicitly
            pricing_key = "gpt-4.1"
        elif "gpt-4-turbo" in model_name.lower() or "gpt-4-1106" in model_name.lower():
            pricing_key = "gpt-4-turbo"
        elif "gpt-4" in model_name.lower():
            pricing_key = "gpt-4"
        elif "gpt-35" in model_name.lower() or "gpt-3.5" in model_name.lower():
            pricing_key = "gpt-35-turbo"
        else:
            pricing_key = "gpt-4"  # Default
        
        pricing = self.PRICING.get(pricing_key, self.PRICING["gpt-4"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def count_and_estimate(
        self,
        input_text: str,
        output_text: str,
        model_name: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Count tokens and estimate cost for input/output pair.
        
        Args:
            input_text: Input text
            output_text: Output text
            model_name: Model name
            
        Returns:
            Dict with token counts and estimated cost
        """
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        estimated_cost = self.estimate_cost(input_tokens, output_tokens, model_name)
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
        }
