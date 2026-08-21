from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-27B"
QUESTION = "Who is the true culprit in Umineko When They Cry?"
MAX_NEW_TOKENS = 8192


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        try:
            output_ids = model.generate(**inputs, **gen_kwargs)
        except TypeError:
            gen_kwargs.pop("presence_penalty")
            output_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
