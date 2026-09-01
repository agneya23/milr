import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaMilrOptimizer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        p_z_dim,
        p_g_dim,
        e_k_dim,
        routing_temperature=1.0,
        text_alpha_max=10.0,
        image_alpha_max=10.0,
        text_trust_radius=1.0,
        image_trust_radius=1.0,
        image_token_cost=1.0,
    ):
        super().__init__()

        self.p_z = nn.Linear(hidden_dim, p_z_dim)
        self.p_g = nn.Linear(hidden_dim, p_g_dim)
        self.modality_embedding = nn.Embedding(2, 4)

        # log-norm, cosine, entropy, position, k/B, reward, delta reward,
        # and remaining budget are the eight scalar token features.
        token_feature_dim = p_z_dim + p_g_dim + 4 + 8
        self.token_encoder = nn.Sequential(
            nn.Linear(token_feature_dim, e_k_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(2 * e_k_dim + 3, 2 * e_k_dim),
            nn.Tanh(),
        )

        self.token_score_text = nn.Linear(e_k_dim, 1)
        self.token_score_image = nn.Linear(e_k_dim, 1)
        self.text_window_bias = nn.Linear(2 * e_k_dim + 1, 1)
        self.image_prefix_bias = nn.Linear(2 * e_k_dim + 1, 1)

        self.h_beta = nn.Linear(3 * e_k_dim, 1)
        self.h_alpha = nn.Linear(3 * e_k_dim, 1)
        self.h_q = nn.Linear(2 * e_k_dim, 1)

        self.text_window_ratios = (0.05, 0.10, 0.20, 0.40)
        self.image_prefix_ratios = (0.005, 0.01, 0.02, 0.04, 0.08)
        self.text_window_stride = 10
        self.routing_temperature = routing_temperature
        self.text_alpha_max = text_alpha_max
        self.image_alpha_max = image_alpha_max
        self.text_trust_radius = text_trust_radius
        self.image_trust_radius = image_trust_radius
        self.image_token_cost = image_token_cost

    @staticmethod
    def _token_view(states):
        # Image latents contain conditional and unconditional CFG branches.
        return states.mean(dim=1) if states.ndim == 3 else states

    def _token_features(
        self,
        z_k_t,
        z_k_i,
        g_k_t,
        g_k_i,
        d_k_t_prev,
        d_k_i_prev,
        text_entropy,
        image_entropy,
        step_index,
        budget,
        reward_value,
        reward_delta,
    ):
        dtype = self.p_z.weight.dtype
        device = z_k_t.device

        z_t = self._token_view(z_k_t).to(dtype)
        z_i = self._token_view(z_k_i).to(dtype)
        g_t = self._token_view(g_k_t).to(dtype)
        g_i = self._token_view(g_k_i).to(dtype)
        d_t = self._token_view(d_k_t_prev).to(dtype)
        d_i = self._token_view(d_k_i_prev).to(dtype)

        z_k = torch.cat((z_t, z_i), dim=0)
        g_k = torch.cat((g_t, g_i), dim=0)
        d_k = torch.cat((d_t, d_i), dim=0)
        num_text = z_t.shape[0]
        num_image = z_i.shape[0]

        z_low = self.p_z(F.layer_norm(z_k, (z_k.shape[-1],)))
        g_low = self.p_g(F.layer_norm(g_k, (g_k.shape[-1],)))
        log_g_norm = torch.log(torch.linalg.vector_norm(g_k, dim=-1) + 1e-8)
        cosine = F.cosine_similarity(g_k, d_k, dim=-1, eps=1e-8)

        entropy = torch.cat((text_entropy, image_entropy), dim=0).to(
            device=device, dtype=dtype
        )
        text_position = torch.linspace(0.0, 1.0, max(num_text, 1), device=device)[:num_text]
        image_position = torch.linspace(0.0, 1.0, max(num_image, 1), device=device)[:num_image]
        position = torch.cat((text_position, image_position), dim=0).to(dtype)

        modality_ids = torch.cat(
            (
                torch.zeros(num_text, dtype=torch.long, device=device),
                torch.ones(num_image, dtype=torch.long, device=device),
            )
        )
        modality = self.modality_embedding(modality_ids)

        token_count = num_text + num_image
        step_fraction = float(step_index) / max(float(budget), 1.0)
        remaining_fraction = float(budget - step_index) / max(float(budget), 1.0)
        shared = z_low.new_tensor(
            [step_fraction, reward_value, reward_delta, remaining_fraction]
        ).view(1, 4)
        shared = shared.expand(token_count, -1)

        x_k = torch.cat(
            (
                z_low,
                g_low,
                log_g_norm.unsqueeze(-1),
                cosine.unsqueeze(-1),
                entropy.unsqueeze(-1),
                position.unsqueeze(-1),
                modality,
                shared,
            ),
            dim=-1,
        )
        return x_k, num_text, num_image

    def _optimizer_state(
        self, x_k, num_text, reward_value, reward_delta, step_index, budget
    ):
        e_k = self.token_encoder(x_k)
        e_k_t = e_k[:num_text]
        e_k_i = e_k[num_text:]
        text_pool = e_k_t.mean(dim=0)
        image_pool = e_k_i.mean(dim=0)
        history = e_k.new_tensor(
            [reward_value, reward_delta, float(step_index) / max(float(budget), 1.0)]
        )
        s_k = self.global_encoder(torch.cat((text_pool, image_pool, history), dim=-1))
        return s_k, e_k_t, e_k_i

    def _choose_candidate(self, masks, ratios, token_scores, state, bias_head, deterministic):
        candidate_masks = torch.stack(masks, dim=0)
        candidate_ratios = token_scores.new_tensor(ratios).unsqueeze(-1)
        denominators = candidate_masks.sum(dim=-1).clamp_min(1.0)
        mean_scores = (candidate_masks * token_scores.unsqueeze(0)).sum(dim=-1)
        mean_scores = mean_scores / denominators
        state_features = state.unsqueeze(0).expand(len(masks), -1)
        scores = mean_scores + bias_head(
            torch.cat((state_features, candidate_ratios), dim=-1)
        ).squeeze(-1)

        probabilities = torch.softmax(scores / self.routing_temperature, dim=0)
        if deterministic:
            choice = F.one_hot(scores.argmax(), num_classes=len(masks)).to(scores.dtype)
        else:
            choice = F.gumbel_softmax(
                scores, tau=self.routing_temperature, hard=True, dim=0
            )
        mask = choice @ candidate_masks
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum()
        return mask, entropy

    def _text_mask(self, e_k_t, state, deterministic):
        num_text = e_k_t.shape[0]
        masks = [e_k_t.new_zeros(num_text)]
        ratios = [0.0]
        for ratio in self.text_window_ratios:
            width = min(num_text, max(1, math.ceil(ratio * num_text)))
            starts = list(range(0, max(num_text - width + 1, 1), self.text_window_stride))
            last_start = max(num_text - width, 0)
            if starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                mask = e_k_t.new_zeros(num_text)
                mask[start : start + width] = 1.0
                masks.append(mask)
                ratios.append(ratio)
        token_scores = self.token_score_text(e_k_t).squeeze(-1)
        return self._choose_candidate(
            masks,
            ratios,
            token_scores,
            state,
            self.text_window_bias,
            deterministic,
        )

    def _image_mask(self, e_k_i, state, deterministic):
        num_image = e_k_i.shape[0]
        masks = [e_k_i.new_zeros(num_image)]
        ratios = [0.0]
        for ratio in self.image_prefix_ratios:
            width = min(num_image, max(1, math.ceil(ratio * num_image)))
            mask = e_k_i.new_zeros(num_image)
            mask[:width] = 1.0
            masks.append(mask)
            ratios.append(ratio)
        token_scores = self.token_score_image(e_k_i).squeeze(-1)
        return self._choose_candidate(
            masks,
            ratios,
            token_scores,
            state,
            self.image_prefix_bias,
            deterministic,
        )

    @staticmethod
    def _normalized_direction(gradient):
        rms = torch.sqrt(torch.mean(gradient.float() ** 2, dim=-1, keepdim=True) + 1e-8)
        return gradient.float() / rms

    @staticmethod
    def _clip_to_radius(update, radius):
        norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
        scale = torch.clamp(radius / (norm + 1e-8), max=1.0)
        return update * scale

    def _learned_update(self, g_k_t, g_k_i, d_k_t_prev, d_k_i_prev, e_k_t, e_k_i, state):
        text_context = torch.cat(
            (e_k_t, state.unsqueeze(0).expand(e_k_t.shape[0], -1)), dim=-1
        )
        image_context = torch.cat(
            (e_k_i, state.unsqueeze(0).expand(e_k_i.shape[0], -1)), dim=-1
        )

        beta_t = torch.sigmoid(self.h_beta(text_context))
        beta_i = torch.sigmoid(self.h_beta(image_context)).unsqueeze(1)
        alpha_t = self.text_alpha_max * torch.sigmoid(self.h_alpha(text_context))
        alpha_i = self.image_alpha_max * torch.sigmoid(self.h_alpha(image_context)).unsqueeze(1)

        g_t_normalized = self._normalized_direction(g_k_t)
        g_i_normalized = self._normalized_direction(g_k_i)
        d_k_t = beta_t * d_k_t_prev.float() + (1.0 - beta_t) * g_t_normalized
        d_k_i = beta_i * d_k_i_prev.float() + (1.0 - beta_i) * g_i_normalized

        update_t = self._clip_to_radius(alpha_t * d_k_t, self.text_trust_radius)
        update_i = self._clip_to_radius(alpha_i * d_k_i, self.image_trust_radius)
        return d_k_t, d_k_i, update_t, update_i

    def forward(
        self,
        z_k_t,
        z_k_i,
        g_k_t,
        g_k_i,
        d_k_t_prev,
        d_k_i_prev,
        text_entropy,
        image_entropy,
        step_index,
        budget,
        reward_value,
        reward_delta,
        optimize_mode="both",
        deterministic=False,
    ):
        x_k, num_text, num_image = self._token_features(
            z_k_t,
            z_k_i,
            g_k_t,
            g_k_i,
            d_k_t_prev,
            d_k_i_prev,
            text_entropy,
            image_entropy,
            step_index,
            budget,
            reward_value,
            reward_delta,
        )
        state, e_k_t, e_k_i = self._optimizer_state(
            x_k, num_text, reward_value, reward_delta, step_index, budget
        )

        if optimize_mode == "image":
            text_mask = e_k_t.new_zeros(num_text)
            text_routing_entropy = e_k_t.new_zeros(())
        else:
            text_mask, text_routing_entropy = self._text_mask(
                e_k_t, state, deterministic
            )
        if optimize_mode == "text":
            image_mask = e_k_i.new_zeros(num_image)
            image_routing_entropy = e_k_i.new_zeros(())
        else:
            image_mask, image_routing_entropy = self._image_mask(
                e_k_i, state, deterministic
            )

        d_k_t, d_k_i, update_t, update_i = self._learned_update(
            g_k_t,
            g_k_i,
            d_k_t_prev,
            d_k_i_prev,
            e_k_t,
            e_k_i,
            state,
        )
        update_t = text_mask.unsqueeze(-1) * update_t
        update_i = image_mask.view(-1, 1, 1) * update_i

        continuation = state.new_ones(())
        z_k_t_next = z_k_t + (continuation * update_t).to(z_k_t.dtype)
        z_k_i_next = z_k_i + (continuation * update_i).to(z_k_i.dtype)

        token_cost = continuation * (
            text_mask.sum() / max(num_text, 1)
            + self.image_token_cost * image_mask.sum() / max(num_image, 1)
        )
        stats = {
            "token_cost": token_cost,
            "step_cost": continuation,
            "routing_entropy": text_routing_entropy + image_routing_entropy,
            "continuation": continuation,
            "text_mask": text_mask,
            "image_mask": image_mask,
        }
        return z_k_t_next, z_k_i_next, d_k_t, d_k_i, stats
