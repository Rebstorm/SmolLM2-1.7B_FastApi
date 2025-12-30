import sys
import threading
from typing import Any, Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class SmolLM:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.set_default_system_prompt: str = "You are a helpful assistant."
        self.max_new_tokens: int = 100
        print(f"Loading model '{model_name}' to {self.device}...")
        try:
            self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModelForCausalLM.from_pretrained(model_name).to(
                self.device
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def update_config(
        self,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        if system_prompt is not None:
            self.set_default_system_prompt = system_prompt
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        print(
            f"Config updated: system_prompt='{self.set_default_system_prompt}', "
            f"max_new_tokens={self.max_new_tokens}"
        )

    def _format_prompt(self, prompt: str) -> Any:
        messages = []
        if self.set_default_system_prompt:
            messages.append(
                {"role": "system", "content": self.set_default_system_prompt}
            )
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        formatted_prompt = self._format_prompt(prompt)
        print(f"Generating response for prompt: {prompt}")
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return str(
            self.tokenizer.decode(
                outputs[0][len(inputs[0]) :], skip_special_tokens=True
            )
        )

    def stream_generate(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        formatted_prompt = self._format_prompt(prompt)
        print(f"Streaming response for prompt: {prompt}")
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            inputs, streamer=streamer, max_new_tokens=max_new_tokens
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text
