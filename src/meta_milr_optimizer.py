import torch
import torch.nn as nn


class MetaMilrOptimizer(nn.Module):

    def __init__(self, hidden_dim, p_z_dim, p_g_dim, e_k_dim):
        super().__init__()

        self.p_z = nn.Linear(hidden_dim, p_z_dim)
        self.p_g = nn.Linear(hidden_dim, p_g_dim)

        self.token_encoder = nn.Linear((p_z_dim + p_g_dim + 4), e_k_dim)

        self.token_score_text = nn.Linear(e_k_dim, 1)
        self.token_score_image = nn.Linear(e_k_dim, 1)
        self.step = 10
        self.b_phi = nn.Linear()
        self.R_t = [0.05, 0.1, 0.2, 0.4]
        self.R_v = [0.005, 0.01, 0.02, 0.04, 0.008]

        self.h_beta = nn.Linear(3 * e_k_dim, 1)
        self.h_alpha = nn.Linear(3 * e_k_dim, 1)

    def get_token_feature_vector(
        self, z_k_t, z_k_i, g_k_t, g_k_i, d_k_t_prev, d_k_i_prev, e_mod, e_pos
    ):
        num_text = z_k_t.shape[0]
        num_image = z_k_i.shape[0]
        z_k = torch.cat((z_k_t, z_k_i), 0)
        z_k_low = self.p_z(z_k)
        g_k = torch.cat((g_k_t, g_k_i), 0)
        g_k_low = self.p_g(g_k)
        g_k_norm = torch.sqrt(torch.sum(g_k**2, -1))
        d_k = torch.cat((d_k_t_prev, d_k_i_prev), 0)
        cos_g_d = torch.sum(g_k * d_k, -1)
        x_k = torch.cat((z_k_low, g_k_low, g_k_norm, cos_g_d, e_mod, e_pos), -1)
        return x_k, num_text, num_image

    def get_opt_state(self, x_k, num_text):
        e_k = self.token_encoder(x_k)
        e_k_t = e_k[:num_text, :]
        e_k_t_pool = torch.mean(e_k_t, 0)
        e_k_i = e_k[num_text:, :]
        e_k_i_pool = torch.mean(e_k_i, 0)
        s_k = torch.cat((e_k_t_pool, e_k_i_pool), -1)
        return s_k, e_k_t, e_k_i

    def get_text_window(self, num_text, e_k_t):
        A_rho = [i for i in range(0, num_text, self.step)]
        candidate_set = []
        for a in A_rho:
            for rho in self.R_t:
                candidate_set.append((a, a + int(rho * num_text) - 1))
        maxx_score = -1e10
        maxx_window = -1
        for start, end in candidate_set:
            if end >= num_text:
                end = num_text - 1
            e_k_t_window = e_k_t[start : end + 1, :]
            token_score_e_k_t_window = self.token_score_text(e_k_t_window)
            text_window_score = torch.sum(token_score_e_k_t_window) / len(
                token_score_e_k_t_window
            )  ### need to add bias self.b_phi
            maxx_window = (
                (start, end) if text_window_score > maxx_score else maxx_window
            )
            maxx_score = max(maxx_score, text_window_score)
        if maxx_score < 0:
            return None
        binary_mask = torch.zeros(num_text, dtype=torch.int)
        binary_mask[maxx_window[0] : maxx_window[1] + 1] = 1
        return binary_mask

    def get_image_prefix(self, num_image, e_k_i):
        candidate_set = []
        for rho in self.R_v:
            candidate_set.append((0, int(rho * num_image) - 1))
        maxx_score = -1e10
        maxx_window = -1
        for start, end in candidate_set:
            if end >= num_image:
                end = num_image - 1
            e_k_i_window = e_k_i[start : end + 1, :]
            token_score_e_k_i_window = self.token_score_image(e_k_i_window)
            image_window_score = torch.sum(token_score_e_k_i_window) / len(
                token_score_e_k_i_window
            )  ### need to add bias self.b_phi
            maxx_window = (
                (start, end) if image_window_score > maxx_score else maxx_window
            )
            maxx_score = max(maxx_score, image_window_score)
        if maxx_score < 0:
            return None
        binary_mask = torch.zeros(num_image, dtype=torch.int)
        binary_mask[maxx_window[0] : maxx_window[1] + 1] = 1
        return binary_mask

    def get_binary_mask(self, num_text, num_image, e_k_t, e_k_i):
        m_t_k, m_v_k = self.get_text_window(num_text, e_k_t), self.get_image_prefix(
            num_image, e_k_i
        )
        m_k = torch.cat((m_t_k, m_v_k), 0)
        return m_k

    def get_learned_update(self):
        pass

    def get_cont_prob(self):
        pass

    def forward(self, **kwargs):

        # Optimizer State Representation
        x_k, num_text, num_image = self.get_token_feature_vector(
            z_k_t, z_k_i, g_k_t, g_k_i, d_k_t_prev, d_k_i_prev, e_mod, e_pos
        )
        s_k, e_k_t, e_k_i = self.get_opt_state(x_k, num_text)

        # Where to Update
        m_k = self.get_binary_mask(num_text, num_image, e_k_t, e_k_i)

        # How to Update
        d_k, del_z_k = self.get_learned_update()

        # When to Update
        q_k = self.get_cont_prob()

        # Make Update
        z_k_t_next = z_k_t + q_k * del_z_k
        z_k_i_next = z_k_i + q_k * del_z_k

        return z_k_t_next, z_k_i_next
