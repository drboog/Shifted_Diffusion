import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import Encoder as TransformerEncoder
from x_transformers import Decoder as TransformerDecoder
from x_transformers import TransformerWrapper
from x_transformers.x_transformers import AbsolutePositionalEmbedding
from model_lib.nn import timestep_embedding, linear
# from model_lib.decoder.modules import LayerNorm
# pip install x-transformers , or from link: https://github.com/lucidrains/x-transformers
import math

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

def make_attention_layers(dim, depth, heads, type='decoder'):
    layers = TransformerEncoder if type=='encoder' else TransformerDecoder
    return layers(  dim = dim,
                    depth = depth,
                    heads = heads,
                    use_scalenorm = True, # set to true to use for all layers
                    use_rmsnorm = True, # set to true to use for all layers
                    ff_glu = True, # set to true to use for all feedforwards
                    ) 


class ClipPrior(nn.Module):
    def __init__(self, xf_width=1024, xf_layers=16, xf_heads=32, clip_width=512*3, learn_sigma=False, t5_dim=1024,
                use_vocab=False, vocab_size=16, vocab_use_mean=True, vocab_sample_num=1,
                vocab_log_std_init=0., vocab_mean_init=0., vocab_learnable=False, vocab_std_scale=4., vocab_lr_scale=1., vocab_exp=False):
        super().__init__()

        self.xf_width = xf_width 

        self.transformer = make_attention_layers( xf_width, xf_layers, xf_heads, "decoder" )

        self.time_embed = nn.Sequential(
            linear(xf_width, xf_width),
            nn.SiLU(),
            linear(xf_width, xf_width),
        )

        self.t5_proj = nn.Sequential(
            LayerNorm(t5_dim),
            linear(t5_dim, xf_width),
            nn.SiLU(),
            LayerNorm(xf_width),
            linear(xf_width, xf_width),
            )

        self.x_proj = nn.Sequential(
            linear(clip_width, xf_width),
            nn.SiLU(),
            LayerNorm(xf_width),
            linear(xf_width, xf_width),
            )

        self.clip_sentence_proj = nn.Sequential(
            linear(clip_width, xf_width),
            nn.SiLU(),
            LayerNorm(xf_width),
            linear(xf_width, xf_width) )

        self.final_emb = nn.Parameter( torch.randn(1, 1, xf_width) )

        out_dim = clip_width if not learn_sigma else clip_width*2
        self.out_proj = nn.Sequential(
            linear(xf_width, out_dim),
            nn.SiLU(),
            linear(out_dim, out_dim),
            )
        if use_vocab:
            self.vocab = Vocab(size=vocab_size, use_mean=vocab_use_mean, dim=clip_width, sample_num=vocab_sample_num,
                                log_std_init=vocab_log_std_init, mean_init=vocab_mean_init, learnable=vocab_learnable,
                                std_scale=vocab_std_scale, lr_scale=vocab_lr_scale, exp=vocab_exp)
            self.pos_embed = nn.Sequential(
            linear(xf_width, xf_width),
            nn.SiLU(),
            linear(xf_width, xf_width),
            )   
        else:
            self.vocab = None

    def reshape_to_seq_last(self, input ):
        channel_dim = [512, 1024, 2048, self.xf_width]
        if len(input.shape)==2:
            return input.unsqueeze(2)
        elif len(input.shape)==3:
            if input.shape[2] in channel_dim:
                return input.permute(0, 2, 1)
            else:
                return input 
        elif len(input.shape)==4:
            if input.shape[1] in channel_dim:
                return input.view( input.shape[0], channel_dim, -1 )
        print(f"wrong input shape: {input.shape}")

    def reshape_to_channel_last(self, input ):
        channel_dim = [512, 1024, 2048, self.xf_width]
        if len(input.shape)==3:
            if input.shape[1] in channel_dim:
                return input.permute(0, 2, 1)
            else:
                return input 
        elif len(input.shape)==2:
            return input.unsqueeze(1)
        print(f"wrong input shape: {input.shape}")

    def forward(self, x, timesteps, t5_word_emb, clip_sentence_emb, emb_4_vocab, indices=None, testing=False):
        # use a B x Seq_len order x C
        B = x.shape[0]

        x = self.reshape_to_channel_last( self.x_proj(x) )

        temb = self.reshape_to_channel_last( self.time_embed( timestep_embedding(timesteps, self.xf_width) ) )
        
        clip_s_emb = self.clip_sentence_proj( self.reshape_to_channel_last( clip_sentence_emb ) )
        text_embeddings = self.t5_proj( self.reshape_to_channel_last( t5_word_emb ) )
        if self.vocab is not None:
            if indices is None:
                indices, _, _ = self.vocab.get_indices(emb_4_vocab,) 
            pos_emb = self.reshape_to_channel_last( self.pos_embed(timestep_embedding(indices.view(timesteps.shape), self.xf_width) ) )
            inp = torch.cat([ text_embeddings, clip_s_emb, temb, pos_emb, x, self.final_emb.repeat(x.shape[0], 1, 1) ], dim=1)
        else:    
            inp = torch.cat([ text_embeddings, clip_s_emb, temb, x, self.final_emb.repeat(x.shape[0], 1, 1) ], dim=1)
        
        out = self.transformer( inp )

        if not testing:
            out = self.out_proj(out[:,-1])
            return out

        choice_1, choice_2 = self.out_proj(out[:,-2]), self.out_proj(out[:,-1])
        print( choice_1.shape, choice_2.shape, clip_sentence_emb.shape )
        results = []
        one_over_two = F.cosine_similarity(choice_1, clip_sentence_emb) > F.cosine_similarity(choice_2, clip_sentence_emb)
        for bid in range(choice_1.shape[0]):
            if one_over_two[bid]:
                print("used 1")
                results.append(choice_1[bid])
            else:
                print("used 2")
                results.append(choice_2[bid])
        return torch.stack(results, dim=0)


class Discriminator(nn.Module):
    def __init__(self, xf_width=1024, xf_layers=16, xf_heads=32, clip_width=512*3, learn_sigma=False,):
        super().__init__()

        self.xf_width = xf_width 

        self.transformer = make_attention_layers( xf_width, xf_layers, xf_heads, "decoder" )

        self.time_embed = nn.Sequential(
            linear(xf_width, xf_width),
            nn.SiLU(),
            linear(xf_width, xf_width),
        )

        self.t5_proj = nn.Sequential(
            LayerNorm(1024),
            linear(1024, xf_width),
            nn.SiLU(),
            LayerNorm(xf_width),
            linear(xf_width, xf_width),
            )

        self.x_proj = nn.Sequential(
            linear(clip_width, xf_width),
            nn.SiLU(),
            LayerNorm(xf_width),
            linear(xf_width, xf_width),
            )

        self.clip_sentence_proj = nn.Sequential(
            linear(clip_width, xf_width),
            nn.SiLU(),
            LayerNorm(xf_width),
            linear(xf_width, xf_width) )

        self.final_emb = nn.Parameter( torch.randn(1, 1, xf_width) )

        out_dim = 1 # output 1-dim logit
        self.out_proj = nn.Sequential(
            linear(xf_width, out_dim),
            )

    def reshape_to_seq_last(self, input ):
        channel_dim = [512, 1024, 2048, self.xf_width]
        if len(input.shape)==2:
            return input.unsqueeze(2)
        elif len(input.shape)==3:
            if input.shape[2] in channel_dim:
                return input.permute(0, 2, 1)
            else:
                return input 
        elif len(input.shape)==4:
            if input.shape[1] in channel_dim:
                return input.view( input.shape[0], channel_dim, -1 )
        print(f"wrong input shape: {input.shape}")

    def reshape_to_channel_last(self, input ):
        channel_dim = [512, 1024, 2048, self.xf_width]
        if len(input.shape)==3:
            if input.shape[1] in channel_dim:
                return input.permute(0, 2, 1)
            else:
                return input 
        elif len(input.shape)==2:
            return input.unsqueeze(1)
        print(f"wrong input shape: {input.shape}")

    def forward(self, x, timesteps, t5_word_emb, clip_sentence_emb, emb_4_vocab, indices=None, testing=False):
        # use a B x Seq_len order x C
        B = x.shape[0]

        x = self.reshape_to_channel_last( self.x_proj(x) )

        temb = self.reshape_to_channel_last( self.time_embed( timestep_embedding(timesteps, self.xf_width, max_period=1e6) ) )
        
        clip_s_emb = self.clip_sentence_proj( self.reshape_to_channel_last( clip_sentence_emb ) )
        text_embeddings = self.t5_proj( self.reshape_to_channel_last( t5_word_emb ) )
        if self.vocab is not None:
            if indices is None:
                indices, _, _ = self.vocab.get_indices(emb_4_vocab,) 
            pos_emb = self.reshape_to_channel_last( self.pos_embed(timestep_embedding(indices, self.xf_width) ) )
            inp = torch.cat([ text_embeddings, clip_s_emb, temb, pos_emb, x, self.final_emb.repeat(x.shape[0], 1, 1) ], dim=1)
        else:    
            inp = torch.cat([ text_embeddings, clip_s_emb, temb, x, self.final_emb.repeat(x.shape[0], 1, 1) ], dim=1)
        out = self.transformer( inp )

        return self.out_proj(out[:,-1]) # output the logits


class Vocab(nn.Module):
    """
    Vocabulary of initialization, contains (learnable) words, 
    which serves as the initialization for diffusion process
    """
    def __init__(self, 
                    mean_init=0.,
                    log_std_init=0.,
                    size=16, 
                    dim=1536, 
                    learnable=False, 
                    std_scale=4., 
                    use_mean=True, 
                    sample_num=1, 
                    lr_scale=1.,
                    exp = False):
        super().__init__()
        self.learnable = learnable
        self.size = size
        self.std_scale = std_scale
        self.lr_scale = lr_scale
        self.log_std_init = log_std_init
        self.mean_init = mean_init
        self.exp = exp
        assert self.log_std_init.shape[0] == 1 or self.log_std_init.shape[0] == size 
        assert self.mean_init.shape[0] == 1 or self.mean_init.shape[0] == size 
        assert self.mean_init.shape[-1] == dim

        self.dim = dim
        self.use_mean = use_mean
        self.sample_num = sample_num

        if learnable:
            self.mean = nn.Embedding(size, self.dim)
            self.mean.weight = nn.Parameter(self.mean_init.clone()/self.lr_scale)

            self.log_std = nn.Embedding(size, self.dim) # this is log std actually
            self.log_std.weight = nn.Parameter(self.log_std_init.clone()/self.lr_scale)

        else:
            self.lr_scale = 1.
            self.mean = self.mean_init.clone()
            if self.size > 1:
                self.std = torch.zeros_like(self.log_std_init) # we use std instead of log std when manully update
            else:
                self.std = self.log_std_init.exp()
            self.n = torch.ones((size, 1)) # stores how many samples have each center seen
            self.diff = self.mean_init.clone()#torch.zeros((size, self.dim)) # a tensor used in updating std

    def get_indices(self, input_emb,):
        use_mean = self.use_mean
        sample_num = self.sample_num

        if self.learnable:
            centers = (self.mean.weight.data * self.lr_scale).to(device=input_emb.device) # get the tensor value of parameter 
            center_std = (self.log_std.weight.data * self.lr_scale).exp().to(device=input_emb.device)
        else:
            centers = self.mean.to(device=input_emb.device)
            center_std = self.std.to(device=input_emb.device)
        # print(centers.shape, center_std.shape)
        assert centers.shape == center_std.shape == (self.size, self.dim)

        if use_mean:
            d = - torch.cosine_similarity(centers.unsqueeze(0), input_emb.unsqueeze(1), dim=-1)
            assert d.shape == (input_emb.shape[0], self.size)

        else:# sample_num > 1:
            samples = centers.unsqueeze(0) + center_std.unsqueeze(0) * torch.randn((sample_num, self.size, self.dim)).to(device=centers.device)
            assert samples.shape == (sample_num, self.size, self.dim)
            d = - torch.cosine_similarity(samples.unsqueeze(1), input_emb.unsqueeze(0).unsqueeze(2), dim=-1)
            d = d.sum(0) # now shape (batch_size, vocab_size)
            assert d.shape == (input_emb.shape[0], self.size)

        indices = torch.argmin(d, dim=-1)
        # print(d)
        # print(indices.min(), indices.max())
        if self.learnable:
            mean_q = self.mean(indices).view(input_emb.shape) * self.lr_scale
            std_q = (self.log_std(indices).view(input_emb.shape) * self.lr_scale).exp() # * self.std_scale
        else:
            mean_q = self.mean[indices].view(input_emb.shape) 
            std_q = self.std[indices].view(input_emb.shape)#.exp() # * self.std_scale
        if self.exp:
            if self.learnable:
                std_q = std_q.exp().detach()/std_q.detach() * std_q
            else:
                std_q = std_q.exp()
        return indices, mean_q, std_q

    def update(self, input_emb, ):
        # update the un-learnable mean and std during sampling
        gather_emb = [torch.zeros_like(input_emb) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gather_emb, input_emb)
        gather_emb = torch.cat(gather_emb).view((-1, self.dim))
        
        input_indices, _, _ = self.get_indices(input_emb, )
        indices = [torch.zeros_like(input_indices) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(indices, input_indices)
        indices = torch.cat(indices)
        
        # indices, _, _ = self.get_indices(gather_emb, )

        indices_mat = F.one_hot(indices, num_classes=self.size).t().float().to(device=input_emb.device)
        assert indices_mat.shape == (self.size, gather_emb.shape[0])
        new_added = torch.matmul(indices_mat, gather_emb).to(device=input_emb.device) # i^th row is the sum of samples assigned to i^th center
        added_num = indices_mat.sum(1).view((-1, 1)).to(device=input_emb.device)

        self.mean = self.mean.to(device=input_emb.device)
        self.n = self.n.to(device=input_emb.device)
        self.diff = self.diff.to(device=input_emb.device)
        self.std = self.std.to(device=input_emb.device)

        new_n = self.n + added_num
        new_mean = (self.mean * self.n + new_added)/new_n
        # print(added_num, new_mean - self.mean)

        new_var = self.std**2 * self.n / new_n + \
            (new_mean - self.mean)**2 * self.n/new_n + \
                torch.matmul(indices_mat, (gather_emb - torch.matmul(indices_mat.t(), new_mean))**2) / new_n +\
                    2 * (self.mean - new_mean) * (self.diff - self.n*self.mean) / new_n
        self.mean = new_mean
        self.std = torch.sqrt(new_var)
        self.n = new_n
        self.diff = self.diff + new_added

