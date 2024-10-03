import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
import math
import warnings
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        # h = int(n ** 0.5)
        # w = int(n ** 0.5)
        h = math.floor(n ** 0.5)
        w = math.floor(n ** 0.5)
        #直接去掉不等于行和列乘积的多余的特征
        if h*w != n:
            print("h*w",h*w,"n",n)
            indices = torch.randint(0, n, (h*w,))  # 在 [0, n) 范围内生成 h*w 个随机整数作为索引
            x_square = x[:, indices, :] # 根据随机索引选择子集
            x=x_square
            b, n, c = x.shape
            # print("x",x.size())

        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2] #[1,1764,1024] # make torchscript happy (cannot use tensor as tuple)

        sliced_q = q[:, :, :].reshape(b, h, w, c).permute(0, 3, 1, 2)
        pooled_q = self.pool(sliced_q)
        reshaped_pooled_q = pooled_q.reshape(b, c, -1)
        agent_tokens = reshaped_pooled_q.permute(0, 2, 1)#[1,49,1024]

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)#[1,8,1764,128]
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)#[1,8,49,128]

        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')#[8,49,14,14]

        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)#[1,8,49,196]
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)##[1,8,49,196]

        position_bias = position_bias1 + position_bias2#[1,8,49,196]

        k_t= k.transpose(-2, -1)
        weighted_scores = (agent_tokens * self.scale) @ k_t
        position_bias_shape = weighted_scores.shape
        position_bias_new = nn.functional.interpolate(position_bias,size=position_bias_shape[2:],
                                           mode='bilinear', align_corners=False)
        weighted_scores_with_bias = weighted_scores + position_bias_new

        agent_attn = self.softmax(weighted_scores_with_bias)

        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)

        q_attn_1 = (q * self.scale) @ agent_tokens.transpose(-2, -1)
        q_attn_shape=q_attn_1.shape
        agent_bias_new = nn.functional.interpolate(agent_bias,size=q_attn_shape[2:],
                                           mode='bilinear', align_corners=False)
        q_attn = self.softmax(q_attn_1 + agent_bias_new)

        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v[:, :, :, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x[:, :, :] = x[:, :, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x) #[1,1764,1024]
        return x


class AgentBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 agent_num=49, window=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AgentAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                   agent_num=agent_num, window=window)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DSCE(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, n_classes=4, top_k=1, clinic_factor=10, fusion="concat"):
        super(DSCE, self).__init__()
        assert n_classes > 2
        self.size_dict = {"small": [1024, 512]}  # 1024+clinical_len
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

        self.classifiers = nn.ModuleList([nn.Linear(size[1], 1) for i in range(n_classes)])
        initialize_weights(self)
        self.top_k = top_k

        self.n_classes = n_classes
        self.fusion = fusion
        self.clinic_factor = clinic_factor
        self.feature_len=1024
        
        assert self.top_k == 1

        self.agent = nn.Sequential(
            *[
                AgentBlock(
                    dim=1024, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0.,
                    attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                    agent_num=49)
            ]
        )
        self.fc_concact = nn.Sequential(
            nn.Linear(self.feature_len+self.clinic_factor, self.feature_len),
            nn.ReLU(),
        )

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, clinic_data, return_features=False):
        device = h.device
        clinic = clinic_data.expand(h.size(0), -1)
        h_clinic = torch.cat((h, clinic), dim=1)
        h=self.fc_concact(h_clinic)
        
        h=h.unsqueeze(0)
        _, n, _ = h.shape
        w = math.floor(n ** 0.5)

        if w*w != n:
            indices = torch.randint(0, n, (w*w,))
            h_square = h[:, indices, :]
            h=h_square

        h_agent=self.agent(h)
        h = self.fc(h_agent)
        h = h.squeeze(0)
        logits = torch.empty(h.size(0), self.n_classes).float().to(device)

        for c in range(self.n_classes):
            if isinstance(self.classifiers, nn.DataParallel):
                logits[:, c] = self.classifiers.module[c](h).squeeze(1)
            else:
                logits[:, c] = self.classifiers[c](h).squeeze(1)

        y_probs = F.softmax(logits, dim=1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat((
            torch.div(m, self.n_classes, rounding_mode='trunc').view(-1, 1),
            torch.div(m % self.n_classes, self.n_classes, rounding_mode='trunc').view(-1, 1)
        ), dim=1).view(-1, 1)

        top_instance = logits[top_indices[0]]
        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict
