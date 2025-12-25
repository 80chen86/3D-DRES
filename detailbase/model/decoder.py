import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask=None, attn_masks=None, pe=None):
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_masks is not None:
            attn_masks = attn_masks.unsqueeze(1).expand(-1, self.nhead, -1, -1).contiguous().flatten(0,1)
        output, weights = self.attn(query,k,v,attn_mask=attn_masks,key_padding_mask=batch_mask,)
        self.dropout(output)
        output = output + query
        self.norm(output)
        return output

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None, output_mask=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        if output_mask is not None:
            output = output_mask * output
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
    
class Decoder(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.d_model = d_model
        self.nhead = nhead
        H = 768
        self.lang_proj = nn.Linear(H, 256)
            
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, batch_mask, sp_mask_features): 
        pred_masks = torch.einsum('bnd,bmd->bnm', query, sp_mask_features)
        
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]    
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None

        return pred_masks, attn_masks

    def prediction_head(self, query, batch_mask, sp_mask_features):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, batch_mask, sp_mask_features)
        return pred_scores, pred_masks, attn_masks
    
    def get_t2o(self, scores, masks, t2os):
        new_scores, new_masks = [], []
        for i in range(len(t2os)):
            ks = t2os[i].keys()
            for k in ks:
                new_scores.append(scores[i][int(k)+1])
                new_masks.append(masks[i][int(k)+1])
        
        new_scores = torch.stack(new_scores)
        new_masks = torch.stack(new_masks)
        return new_scores, new_masks
    
    def forward_iter_pred(self, x, batch_offsets, lang_feats=None, lang_masks=None, sp_coords_float=None, t2os=None):
        """
        x [B*M, inchannel]
        """
            
        lang_feats = self.lang_proj(lang_feats)   
        lang_masks = ~(lang_masks.bool())
        
        query = lang_feats
        
        inst_feats = self.input_proj(x)
        
        mask_feats = self.x_mask(x)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)

        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        prediction_masks = []
        prediction_scores = []
        prediction_pos = []
        
        pred_scores, pred_masks, attn_masks = self.prediction_head(query, batch_mask, mask_feats)
        pred_scores, pred_masks = self.get_t2o(pred_scores, pred_masks, t2os)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)
        
        # multi-round
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_mask, attn_masks)
            query = self.self_attn_layers[i](query, lang_masks)
            
            query = self.ffn_layers[i](query)
            
            pred_scores, pred_masks, attn_masks = self.prediction_head(query, batch_mask, mask_feats)
            pred_scores, pred_masks = self.get_t2o(pred_scores, pred_masks, t2os)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        
        return {
            'masks':
            pred_masks,
            'lang_masks':
            lang_masks,
            'batch_mask':
            batch_mask,
            'scores':
            pred_scores,
            'aux_outputs': [{
                'masks': a,
                'scores': b,
            } for a, b in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
        }

    def forward(self, x, batch_offsets, lang_feats=None, lang_masks=None, sp_coords_float=None, t2os=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, batch_offsets, lang_feats, lang_masks, sp_coords_float, t2os)
        else:
            raise NotImplementedError
