"""Bind the explicit Qwen3 non-thinking choice for the kwargs-free dataloader."""


def bind_nonthinking(tokenizer):
    tokenizer.chat_template = "{%- set enable_thinking = false %}\n" + tokenizer.chat_template
    messages = [{"role": "user", "content": "Template verification."}]
    # This call deliberately matches RLHFDataset: no thinking keyword is passed.
    actual = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    explicit = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    if actual != explicit or not actual.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        raise ValueError("The dataloader does not render Qwen3's explicit non-thinking template")
    return tokenizer
