import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Utiliser un GPU NVIDIA compatible s'il est disponible, sinon le CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Chargement du modèle sur : {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model = model.to(device)
model.eval()


def generer_reponse(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Tu es un assistant utile. Réponds en français.",
        },
        {"role": "user", "content": question},
    ]

    texte = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    entrees = tokenizer(
        texte,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        sortie = model.generate(
            **entrees,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Retirer la question pour ne garder que les nouveaux tokens.
    nouveaux_tokens = sortie[0, entrees["input_ids"].shape[1]:]

    return tokenizer.decode(
        nouveaux_tokens,
        skip_special_tokens=True,
    ).strip()


def main():
    print("Modèle prêt. Tape 'quitter' pour terminer.")

    while True:
        question = input("\nToi : ").strip()

        if question.lower() in {"quitter", "exit", "quit"}:
            break

        if not question:
            continue

        reponse = generer_reponse(question)
        print(f"\nIA : {reponse}")


if __name__ == "__main__":
    main()
