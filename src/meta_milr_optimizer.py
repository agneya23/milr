import torch
import torch.nn as nn

class MetaMilrOptimizer(nn.Module):

    def __init__(self, hidden_dim, p_z_dim, p_g_dim, e_k_dim):
        super().__init__()

        self.p_z = nn.Linear(hidden_dim, p_z_dim)
        self.p_g = nn.Linear(hidden_dim, p_g_dim)

        self.token_encoder = nn.Linear((p_z_dim+p_g_dim+5), e_k_dim)

        self.token_score_text = nn.Linear(e_k_dim, 1)
        self.b_phi = nn.Linear()
        self.R_t = [0.05, 0.1, 0.2, 0.4]

        self.h_beta = nn.Linear(3*e_k_dim, 1)
        self.h_alpha = nn.Linear(3*e_k_dim, 1)

    def get_token_feature_vector(self, z_k_t, z_k_i, g_k_t, g_k_i, d_k_prev, e_mod, e_pos):
        
        return x_k

    def get_opt_state(self, x_k):
        e_k = self.token_encoder(x_k)
        s_k = torch.cat()
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
        x_k = self.get_token_feature_vector(z_k_t, z_k_i, g_k_t, g_k_i, d_k_prev)
        s_k = self.get_opt_state(x_k)

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