import torch
import torch.nn as nn

class MetaMilrOptimizer(nn.Module):

    def __init__(self, hidden_dim, p_z_dim, p_g_dim, e_k_dim):
        super().__init__()

        self.p_z = nn.Linear(hidden_dim, p_z_dim)
        self.p_g = nn.Linear(hidden_dim, p_g_dim)

        self.token_encoder = nn.Linear((p_z_dim+p_g_dim+4), e_k_dim)

        self.token_score_text = nn.Linear(e_k_dim, 1)
        self.b_phi = nn.Linear()
        self.R_t = [0.05, 0.1, 0.2, 0.4]

        self.h_beta = nn.Linear(3*e_k_dim, 1)
        self.h_alpha = nn.Linear(3*e_k_dim, 1)

    def get_token_feature_vector(self, z_k_t, z_k_i, g_k_t, g_k_i, d_k_t_prev, d_k_i_prev, e_mod, e_pos):
        num_text = z_k_t.shape[0]
        z_k = torch.cat((z_k_t, z_k_i), 0)
        z_k_low = self.p_z(z_k)
        g_k = torch.cat((g_k_t, g_k_i), 0)
        g_k_low = self.p_g(g_k)
        g_k_norm = torch.sqrt(torch.sum(g_k**2, -1))
        d_k = torch.cat((d_k_t_prev, d_k_i_prev), 0)
        cos_g_d = torch.sum(g_k * d_k, -1)
        x_k = torch.cat((z_k_low, g_k_low, g_k_norm, cos_g_d, e_mod, e_pos), -1)
        return x_k, num_text

    def get_opt_state(self, x_k, num_text):
        e_k = self.token_encoder(x_k)
        e_k_t = e_k[:num_text, :]
        e_k_t_pool = torch.mean(e_k_t, 0)
        e_k_i = e_k[num_text:, :]
        e_k_i_pool = torch.mean(e_k_i, 0)
        s_k = torch.cat((e_k_t_pool, e_k_i_pool), -1)
        return s_k

    def get_text_window(self):
        A_rho = []
        candidate_set = 

    def get_image_prefix(self):
        pass

    def get_binary_mask(self):
        m_t_k, m_v_k = self.get_text_window(), self.get_image_prefix()
        m_k = torch.cat()
        return m_k

    def get_learned_update(self):
        pass

    def get_cont_prob(self):
        pass

    def forward(self, **kwargs):

        # Optimizer State Representation
        x_k, num_text = self.get_token_feature_vector(z_k_t, z_k_i, g_k_t, g_k_i, d_k_t_prev, d_k_i_prev, e_mod, e_pos)
        s_k = self.get_opt_state(x_k, num_text)

        # Where to Update
        m_k = self.get_binary_mask()

        # How to Update
        d_k, del_z_k = self.get_learned_update()

        # When to Update
        q_k = self.get_cont_prob()

        # Make Update
        z_k_t_next = z_k_t + q_k * del_z_k
        z_k_i_next = z_k_i + q_k * del_z_k

        return z_k_t_next, z_k_i_next