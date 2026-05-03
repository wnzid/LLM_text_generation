import json
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_NAME = "gpt2"
INPUT_PHRASE = "Life in Lithuania is"
MAX_NEW_TOKENS = 25
SEED = 42
OUTPUT_FILE = Path("generation_results.json")

def setup_seed(seed: int = 42) -> None:
    """Make sampling results more repeatable."""
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name: str):
    """Load tokenizer and causal language model from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def decode_output(tokenizer, output_ids) -> str:
    """Convert generated token IDs back into normal text."""
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_all_methods(tokenizer, model, device: str, input_phrase: str) -> dict:
    """Generate text using all required methods."""
    input_ids = tokenizer(input_phrase, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    results = {}

    #1. Deterministic: Greedy decoding
    with torch.no_grad():
        greedy_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Greedy decoding"] = decode_output(tokenizer, greedy_ids)

    #2. Deterministic: Beam search
    with torch.no_grad():
        beam_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=5,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Beam search, 5 beams"] = decode_output(tokenizer, beam_ids)

    #3. Beam search with repetition penalty
    with torch.no_grad():
        repetition_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=5,
            do_sample=False,
            repetition_penalty=1.2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Beam search + repetition penalty 1.2"] = decode_output(tokenizer, repetition_ids)

    #4. Sampling with low temperature
    with torch.no_grad():
        temp_low_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Temperature sampling, temperature 0.7"] = decode_output(tokenizer, temp_low_ids)

    #5. Sampling with higher temperature
    with torch.no_grad():
        temp_high_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.2,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Temperature sampling, temperature 1.2"] = decode_output(tokenizer, temp_high_ids)

    #6. Top-K sampling
    with torch.no_grad():
        top_k_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_k=40,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Top-K sampling, k=40"] = decode_output(tokenizer, top_k_ids)

    #7. Top-P / nucleus sampling
    with torch.no_grad():
        top_p_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_k=0,
            top_p=0.92,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    results["Top-P / nucleus sampling, p=0.92"] = decode_output(tokenizer, top_p_ids)
    return results

def show_token_information(tokenizer, model, device: str, input_phrase: str) -> None:
    """Optional educational section: show token IDs and most likely next-token candidates."""
    input_ids = tokenizer(input_phrase, return_tensors="pt").input_ids.to(device)

    print("\nInput phrase:", input_phrase)
    print("Token IDs:", input_ids[0].tolist())
    print("Decoded tokens:")
    for token_id in input_ids[0]:
        print(f"  {int(token_id):<6} -> {repr(tokenizer.decode(token_id))}")

    with torch.no_grad():
        final_logits = model(input_ids).logits[0, -1]
        probabilities = torch.softmax(final_logits, dim=0)
        top10 = torch.topk(probabilities, 10)

    print("\nTop 10 possible next tokens:")
    for value, index in zip(top10.values, top10.indices):
        token_text = tokenizer.decode(index)
        print(f"  {repr(token_text):<15} probability={value.item():.2%}")

def main() -> None:
    setup_seed(SEED)
    tokenizer, model, device = load_model_and_tokenizer(MODEL_NAME)

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")

    show_token_information(tokenizer, model, device, INPUT_PHRASE)

    results = generate_all_methods(tokenizer, model, device, INPUT_PHRASE)

    print("\nGenerated outputs:\n")
    for method, text in results.items():
        print("=" * 80)
        print(method)
        print("-" * 80)
        print(text)
        print()

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
