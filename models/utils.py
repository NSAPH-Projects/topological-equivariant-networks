import torch
import torch.nn as nn
from torch_scatter import scatter_add


class MessageLayer(nn.Module):
    def __init__(self, num_hidden, num_inv):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * num_hidden + num_inv, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU()
        )
        self.edge_inf_mlp = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x, index, edge_attr):
        index_send, index_rec = index
        x_send, x_rec = x
        sim_send, sim_rec = x_send[index_send], x_rec[index_rec]
        state = torch.cat((sim_send, sim_rec, edge_attr), dim=1)

        messages = self.message_mlp(state)
        edge_weights = self.edge_inf_mlp(messages)
        messages_aggr = scatter_add(messages * edge_weights, index_rec, dim=0)

        return messages_aggr


class UpdateLayer(nn.Module):
    def __init__(self, num_hidden, num_mes):
        super().__init__()
        self.update_mlp = nn.Sequential(
            nn.Linear((num_mes + 1) * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
        )

    def forward(self, x, bound_mes, upadj_mes):
        state = x

        if torch.is_tensor(bound_mes):
            state = torch.cat((state, bound_mes), dim=1)

        if torch.is_tensor(upadj_mes):
            state = torch.cat((state, upadj_mes), dim=1)

        update = self.update_mlp(state)
        return update

def compute_invariants_3d(feat_ind, pos, adj, inv_ind, device):
    # angles
    angle = {}

    vecs = pos[feat_ind['1'][:, 0]] - pos[feat_ind['1'][:, 1]]
    send_vec, rec_vec = vecs[adj['1_1'][0]], vecs[adj['1_1'][1]]
    send_norm, rec_norm = torch.linalg.norm(send_vec, ord=2, dim=1), torch.linalg.norm(rec_vec, ord=2, dim=1)

    dot = torch.sum(send_vec * rec_vec, dim=1)
    cos_angle = dot / (send_norm * rec_norm)
    eps = 1e-6
    angle['1_1'] = torch.arccos(cos_angle.clamp(-1 + eps, 1 - eps)).unsqueeze(1)

    p1, p2, a = pos[inv_ind['1_2'][0]], pos[inv_ind['1_2'][1]], pos[inv_ind['1_2'][2]]
    v1, v2, b = p1 - a, p2 - a, p1 - p2
    v1_n, v2_n, b_n = torch.linalg.norm(v1, dim=1), torch.linalg.norm(v2, dim=1), torch.linalg.norm(b, dim=1)
    v1_a = torch.arccos((torch.sum(v1 * b, dim=1) / (v1_n * b_n)).clamp(-1 + eps, 1 - eps))
    v2_a = torch.arccos((torch.sum(v2 * b, dim=1) / (v2_n * b_n)).clamp(-1 + eps, 1 - eps))
    b_a = torch.arccos((torch.sum(v1 * v2, dim=1) / (v1_n * v2_n)).clamp(-1 + eps, 1 - eps))

    angle['1_2'] = torch.moveaxis(torch.vstack((v1_a + v2_a, b_a)), 0, 1)

    # areas
    area = {}
    area['0'] = torch.zeros(len(feat_ind['0'])).unsqueeze(1)
    area['1'] = torch.norm(pos[feat_ind['1'][:, 0]] - pos[feat_ind['1'][:, 1]], dim=1).unsqueeze(1)
    area['2'] = (torch.norm(torch.cross(pos[feat_ind['2'][:, 0]] - pos[feat_ind['2'][:, 1]],
                                        pos[feat_ind['2'][:, 0]] - pos[feat_ind['2'][:, 2]], dim=1),
                            dim=1) / 2).unsqueeze(1)


    area = {k: v.to(feat_ind['0'].device) for k, v in area.items()}


    inv = {
        '0_0': torch.linalg.norm(pos[adj['0_0'][0]] - pos[adj['0_0'][1]], dim=1).unsqueeze(1),
        '0_1': torch.linalg.norm(pos[inv_ind['0_1'][0]] - pos[inv_ind['0_1'][1]], dim=1).unsqueeze(1),
        '1_1': torch.stack([
            torch.linalg.norm(pos[inv_ind['1_1'][0]] - pos[inv_ind['1_1'][1]], dim=1),
            torch.linalg.norm(pos[inv_ind['1_1'][0]] - pos[inv_ind['1_1'][2]], dim=1),
            torch.linalg.norm(pos[inv_ind['1_1'][1]] - pos[inv_ind['1_1'][2]], dim=1),
        ], dim=1),
        '1_2': torch.stack([
            torch.linalg.norm(pos[inv_ind['1_2'][0]] - pos[inv_ind['1_2'][2]], dim=1)
            + torch.linalg.norm(pos[inv_ind['1_2'][1]] - pos[inv_ind['1_2'][2]], dim=1),
            torch.linalg.norm(pos[inv_ind['1_2'][1]] - pos[inv_ind['1_2'][2]], dim=1)
        ], dim=1),
    }

    for k, v in inv.items():
        area_send, area_rec = area[k[0]], area[k[2]]
        send, rec = adj[k]
        area_send, area_rec = area_send[send], area_rec[rec]
        inv[k] = torch.cat((v, area_send, area_rec), dim=1)

    inv['1_1'] = torch.cat((inv['1_1'], angle['1_1'].to(feat_ind['0'].device)), dim=1)
    inv['1_2'] = torch.cat((inv['1_2'], angle['1_2'].to(feat_ind['0'].device)), dim=1)

    return inv