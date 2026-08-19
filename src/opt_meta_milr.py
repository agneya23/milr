import os
import torch
import numpy as np
import PIL.Image

@torch.inference_mode()
def generate_image_from_prompt(
    mmgpt,
    vl_chat_processor,
    user_prompt: str,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    save_path: str = None
):
    print("user_prompt:", user_prompt)
    
    # === construct chat prompt ===
    conversation = [
        {"role": "<|User|>", "content": user_prompt},
        {"role": "<|Assistant|>", "content": ""}
    ]
    prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=""
    ) + vl_chat_processor.image_start_tag

    # === Tokenize ===
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    parallel_size = 1  # only generate one image
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id  # unconditional

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()

    outputs = None
    for i in range(image_token_num):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i > 0 else None,
        )
        hidden_states = outputs.last_hidden_state  # [2, seq, hidden]

        logits = mmgpt.gen_head(hidden_states[:, -1, :])  # [2, vocab]
        logit_cond = logits[0::2]
        logit_uncond = logits[1::2]
        fused_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(fused_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated_tokens[:, i] = next_token.squeeze(-1)

        next_token = torch.cat([next_token, next_token], dim=1).view(-1)  # [2]
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)

    # === image decoding ===
    decoded = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)

    image = PIL.Image.fromarray(decoded[0])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)

    return image


def support_rollout(**kwargs):

    model, text_hidden_states, image_hidden_states, device, cfg_weight, num_support, temperature, img_size, patch_size, data, reward_model = kwargs["model"], kwargs["text_hidden_states"], kwargs["image_hidden_states"], kwargs["device"], kwargs["cfg_weight"], kwargs["num_support"], kwargs["temperature"], kwargs["img_size"], kwargs["patch_size"], kwargs["data"], kwargs["reward_model"]

    rewards, text_grads, image_grads = [], [], []

    for s in range(num_support):

        generated_image_tokens = torch.zeros((1, len(image_hidden_states)), dtype=torch.int).to(device)

        text_logits = model.language_model.lm_head(text_hidden_states)
        text_probs = torch.softmax(text_logits, dim=-1) + 1e-8
        text_token_ids = torch.argmax(text_probs, dim=-1)
        text_log_pi = torch.log(text_probs[torch.arange(len(text_hidden_states)), 0, text_token_ids]+1e-10)

        image_logits = model.gen_head(image_hidden_states)
        image_logits_cond = image_logits[:, 0, :]
        image_logits_uncond = image_logits[:, 1, :]
        image_fused_logits = image_logits_uncond+cfg_weight*(image_logits_cond-image_logits_uncond)
        image_probs = torch.softmax(image_fused_logits/temperature, dim=-1)
        image_token_ids = torch.multinomial(image_probs, num_samples=1).squeeze(-1)
        generated_image_tokens[:, :] = image_token_ids
        image_log_pi = torch.log(image_probs[torch.arange(len(image_hidden_states)), image_token_ids]+1e-10)

        decoded = model.gen_vision_model.decode_code(
            generated_image_tokens.to(dtype=torch.int), 
            shape=[1, 8, img_size // patch_size, img_size // patch_size]
            )
        decoded = decoded.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)
        new_img = PIL.Image.fromarray(decoded[0])

        reward = reward_model.get_reward(new_img, data)
        text_loss, image_loss = text_log_pi.sum(), image_log_pi.sum()

        total_loss = text_loss+image_loss
        inputs_dict = {"text_hidden_states" : text_hidden_states, "image_hidden_states" : image_hidden_states}
        grads = torch.autograd.grad(total_loss, inputs_dict)

        rewards.append(reward)
        text_grads.append(grads["text_hidden_states"])
        image_grads.append(grads["image_hidden_states"])

    avg_reward, std_reward = sum(rewards)/len(rewards), np.std(rewards)

    g_k_t = torch.zeros_like(text_hidden_states).to(device)
    g_k_i = torch.zeros_like(image_hidden_states).to(device)
    for r, tg, ig in zip(rewards, text_grads, image_grads):
        g_k_t += ((r-avg_reward)/std_reward) * tg
        g_k_i += ((r-avg_reward)/std_reward) * ig
    g_k_t /= num_support # []
    g_k_i /= num_support # []

    return g_k_t, g_k_i


def meta_milr_optimized_generation(**kwargs):
    reward_history = []
    initial_reward = reward_model.get_reward(image, data)
    reward_history.append(initial_reward)

    text_hidden_states, image_hidden_states = torch.tensor(text_hidden_states_list).to(device), torch.tensor(image_hidden_states_list).to(device)

    for i in range(budget): # Proposal Algorithm line 8

        g_k_t, g_k_i = support_rollout(
            **kwargs, 
            text_hidden_states=text_hidden_states, 
            image_hidden_states=image_hidden_states
            )

        meta_milr_optimizer(
            z_k_t=text_hidden_states,
            z_k_i=image_hidden_states,
            g_k_t=g_k_t,
            g_k_i=g_k_i
            )

    return new_img, reward_history, total_img+total, image_token_num+len(generated_seq), img_update_length+update_length