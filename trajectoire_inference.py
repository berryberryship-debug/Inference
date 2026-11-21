#!/usr/bin/env python3
"""Extraction de la trajectoire d'états cachés et du KV Cache lors de l'inférence LLM."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = Path("resultats/trajectoire_inference")
MAX_NEW_TOKENS = 128


def calculer_kv_cache_octets(config, nb_tokens: int, octets_par_param: int) -> int:
    """Formule Inference Engineering : 2 * L * T * H_kv * D_head * b."""
    L = config.num_hidden_layers
    H_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    D_head = config.hidden_size // config.num_attention_heads
    return 2 * L * nb_tokens * H_kv * D_head * octets_par_param


def observer_generation(model, tokenizer, device: str, question: str, octets_param: int):
    messages = [
        {"role": "system", "content": "Tu es un assistant utile. Réponds en français."},
        {"role": "user", "content": question},
    ]
    texte = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    entrees = tokenizer(texte, return_tensors="pt").to(device)
    input_ids = entrees["input_ids"]
    longueur_prompt = input_ids.shape[1]

    temps_cumule = []
    itl_liste = []
    kv_octets_liste = []
    etats_derniere_couche = []
    tokens_generes = []

    past_key_values = None
    cur_ids = input_ids
    t_debut = time.perf_counter()
    t_precedent = t_debut

    with torch.inference_mode():
        for step in range(MAX_NEW_TOKENS):
            sorties = model(
                input_ids=cur_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            maintenant = time.perf_counter()

            # Temps : step 0 = TTFT (Prefill), steps suivants = ITL (Decode)
            delta = maintenant - t_precedent
            t_precedent = maintenant
            temps_cumule.append(maintenant - t_debut)
            itl_liste.append(delta)

            # État interne de la dernière couche au dernier token (vecteur de dimension hidden_size)
            dernier_etat = sorties.hidden_states[-1][0, -1, :].detach().cpu().numpy()
            etats_derniere_couche.append(dernier_etat)

            # Mise à jour du KV Cache et mesure mémoire
            past_key_values = sorties.past_key_values
            total_tokens = longueur_prompt + step + 1
            kv_octets_liste.append(
                calculer_kv_cache_octets(model.config, total_tokens, octets_param)
            )

            # Décodage glouton (do_sample=False)
            prochain_token = torch.argmax(sorties.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(prochain_token.item())
            tokens_generes.append(token_id)

            if token_id == tokenizer.eos_token_id:
                break

            cur_ids = prochain_token

    reponse = tokenizer.decode(tokens_generes, skip_special_tokens=True).strip()
    return (
        reponse,
        np.array(temps_cumule),
        np.array(itl_liste),
        np.array(kv_octets_liste),
        np.array(etats_derniere_couche),
        longueur_prompt,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    octets_param = 4  # FP32 par défaut sur CPU

    print(f"=== Trajectoire d'inférence LLM ({device}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    question = input("\nQuestion à observer : ").strip()
    if not question:
        question = "Explique brièvement ce qu'est un attracteur dans un système dynamique."

    (
        reponse,
        temps,
        itl,
        kv_octets,
        etats,
        longueur_prompt,
    ) = observer_generation(model, tokenizer, device, question, octets_param)

    norme_etats = np.linalg.norm(etats, axis=1)
    variance_etats = etats.var(axis=1)
    # Dérive pas-à-pas dans l'espace latent (distance euclidienne entre t et t-1)
    derive_pas = np.zeros(len(etats))
    if len(etats) > 1:
        derive_pas[1:] = np.linalg.norm(np.diff(etats, axis=0), axis=1)

    np.savez_compressed(
        OUT / "trajectoire_llm.npz",
        temps=temps,
        itl=itl,
        kv_octets=kv_octets,
        etats=etats,
        norme_etats=norme_etats,
        derive_pas=derive_pas,
    )

    with (OUT / "mesures_inference.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pas_token", "temps_cumule_s", "latence_pas_s",
            "kv_cache_ko", "norme_etat", "variance_etat", "derive_pas"
        ])
        for i in range(len(temps)):
            writer.writerow([
                i, f"{temps[i]:.6f}", f"{itl[i]:.6f}",
                f"{kv_octets[i] / 1024:.2f}", f"{norme_etats[i]:.6f}",
                f"{variance_etats[i]:.6f}", f"{derive_pas[i]:.6f}"
            ])

    rapport = {
        "modele": MODEL_ID,
        "device": device,
        "longueur_prompt_tokens": longueur_prompt,
        "tokens_generes": len(temps),
        "ttft_s": float(itl[0]),
        "tpot_moyen_s": float(itl[1:].mean()) if len(itl) > 1 else 0.0,
        "kv_cache_final_ko": float(kv_octets[-1] / 1024),
        "derive_pas_moyenne": float(derive_pas[1:].mean()) if len(derive_pas) > 1 else 0.0,
        "question": question,
        "reponse": reponse,
    }

    with (OUT / "resume.json").open("w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    print(f"\nIA : {reponse}")
    print(f"\nTTFT (Prefill) : {rapport['ttft_s']:.3f} s")
    print(f"TPOT moyen (Decode) : {rapport['tpot_moyen_s']:.3f} s/token")
    print(f"KV Cache final : {rapport['kv_cache_final_ko']:.1f} Ko")
    print(f"Termine. Resultats : {OUT}")


if __name__ == "__main__":
    main()
