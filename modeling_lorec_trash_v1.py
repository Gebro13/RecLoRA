import os
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from transformers import set_seed
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from typing import List, Optional, Tuple, Union
import math
from functools import partial
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from ctr_base.models import BaseModel
from ctr_base.layers import MLPBlock
from utils import weight_init


def reset_lora_parameters(self):
    # initialize A the same way as the default for nn.Linear and B to zero
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)


def lorec_hook(lorec_layer, module, inp, outp):
    # print(inp)
    # print(outp)
    # print(module.training)
    lorec_outp = lorec_layer(inp[0])
    modified_outp = outp + lorec_outp
    # return modified_outp
    outp.copy_(modified_outp)
    # print('yes')
    # print(module)


class Loralayer(nn.Module):
    def __init__(self, lora_config, input_dim=5120,output_dim=5120):
        super().__init__()
        alpha = lora_config.lora_alpha
        r = lora_config.r
        self.lora_A = nn.Linear(input_dim, r, bias=False)
        self.lora_B = nn.Linear(r, output_dim, bias=False) 
        self.lora_dropout = nn.Dropout(p=lora_config.lora_dropout)
        self.scaling = alpha / r

    def reset_lora_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self,x):
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling



class Loreclayer(nn.Module):
    def __init__(self, lora_config, ctr_out_size, ctr_final_dim, input_dim=5120, output_dim=5120,initialization_method = 'kaiming',temperature = 1, which_in_gate = 'None'):
        super().__init__()
        self.lora_config = lora_config
        self.r = lora_config.r
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = lora_config.lora_alpha
        self.ctr_final_dim = ctr_final_dim
        self.initialization_method = initialization_method
        self.temperature = temperature
        self.which_in_gate = which_in_gate
        
        self.ctr_hidden_states = None
        if self.which_in_gate == 'token':
            self.gating_final_layer = nn.Sequential(
                nn.LayerNorm(ctr_out_size+self.input_dim),
                nn.Linear(ctr_out_size+self.input_dim,self.ctr_final_dim),
            )
        elif self.which_in_gate == 'lora':
            self.key_layer = nn.Sequential(
                nn.Linear(2*self.input_dim*self.r,8),
                nn.ReLU(),
                nn.Linear(8,ctr_out_size),
            )
            self.query_layer = nn.Linear(ctr_out_size,ctr_out_size)
            self.attention_layer = nn.Sequential(*[
            MLPBlock(
                input_dim=ctr_out_size * 4, 
                hidden_size=ctr_out_size*4, 
                num_hidden_layers=2, 
                hidden_dropout_rate=0.1, 
                hidden_act='relu', 
            ),
            nn.Linear(in_features=ctr_out_size * 4, out_features=1)
            ])
        else:
            self.ctr_final_layer = nn.Sequential(
                nn.LayerNorm(ctr_out_size),
                nn.Linear(ctr_out_size, 60),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(60,self.ctr_final_dim),
            )
            # self.ctr_final_layer = nn.Sequential(
            #     nn.LayerNorm(ctr_out_size),
            #     nn.Linear(ctr_out_size, self.ctr_final_dim),
            # )
        self.lora_A_adapter = nn.Linear(self.ctr_final_dim, self.input_dim * self.r, bias=False)
        self.lora_B_adapter = nn.Linear(self.ctr_final_dim, self.r * self.output_dim, bias=False)
        self.lora_dropout = nn.Dropout(p=lora_config.lora_dropout)
        self.scaling = self.alpha / self.r

    def reset_lorec_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        if self.which_in_gate == 'token':
            nn.init.kaiming_uniform_(self.gating_final_layer[1].weight, a=math.sqrt(5))
            nn.init.zeros_(self.gating_final_layer[1].bias)
        elif self.which_in_gate == 'lora':
            pass
        else:
            nn.init.kaiming_uniform_(self.ctr_final_layer[1].weight, a=math.sqrt(5))
            nn.init.zeros_(self.ctr_final_layer[1].bias)
            nn.init.kaiming_uniform_(self.ctr_final_layer[-1].weight, a=math.sqrt(5))
            nn.init.zeros_(self.ctr_final_layer[-1].bias)
        if self.initialization_method == 'kaiming':
            for i in range(self.ctr_final_dim):
                nn.init.kaiming_uniform_(self.lora_A_adapter.weight[:, i].reshape(self.r, self.input_dim), a=math.sqrt(5))
        else:
            for i in range(self.ctr_final_dim):
                nn.init.xavier_uniform_(self.lora_A_adapter.weight[:,i].reshape(self.r, self.input_dim))

        nn.init.zeros_(self.lora_B_adapter.weight)

    def forward(self, x):
        bs,seq_len,dim = x.shape

        if self.which_in_gate == 'token':
            ctr_hidden_states_expanded = torch.unsqueeze(self.ctr_hidden_states,1).expand(-1,seq_len,-1)
            ctr_final_states = self.gating_final_layer(torch.cat([x,ctr_hidden_states_expanded],dim=-1))
        elif self.which_in_gate == 'lora':
            lora_weights = torch.cat([self.lora_A_adapter.weight,self.lora_B_adapter.weight],dim=0)
            att_key = self.key_layer(lora_weights.t()) #8*ctr_out_size
            att_query = self.query_layer(self.ctr_hidden_states)

            # ctr_final_states = torch.matmul(att_query, att_key.t())
            
            att_query = att_query.unsqueeze(1).expand(-1, self.ctr_final_dim, -1) #bs*8*ctr_out_size
            att_key = att_key.unsqueeze(0).expand(bs, -1, -1) #bs*8*ctr_out_size
            attention_input = torch.cat([
                att_query, att_key,
                att_query - att_key,
                att_query * att_key,], 
                dim=-1,
            )
            attention_output = self.attention_layer(attention_input).squeeze(-1) # bs * 8
            ctr_final_states = attention_output - torch.max(attention_output, dim=-1, keepdim=True).values

        else:
            ctr_final_states = self.ctr_final_layer(self.ctr_hidden_states)

        if self.lora_config.enable_softmax:
            if self.lora_config.routing_mode == 'soft':
                ctr_final_states = ctr_final_states/self.temperature
            elif self.lora_config.routing_mode == 'hard':
                _, indices = torch.topk(ctr_final_states,self.lora_config.topk)
                ctr_final_states[~indices] = 0
            ctr_final_states = F.softmax(ctr_final_states, dim=-1)
            if self.lora_config.enable_sqrt_after_softmax:
                ctr_final_states = torch.sqrt(ctr_final_states)
            # print(ctr_final_states)
            # print(torch.max(ctr_final_states,dim=1))
        
        if self.which_in_gate == 'token':
            lora_A_weights = self.lora_A_adapter(ctr_final_states).reshape(-1, seq_len, self.r, self.input_dim).permute(0, 1, 3, 2)
            lora_B_weights = self.lora_B_adapter(ctr_final_states).reshape(-1, seq_len, self.r, self.output_dim)
            return torch.matmul(torch.matmul(self.lora_dropout(x).unsqueeze(-2), lora_A_weights), lora_B_weights).squeeze(-2) * self.scaling
        else:
            # ctr_final_states = (torch.ones(ctr_final_states.shape)/8).to(ctr_final_states.device)
            lora_A_weights = self.lora_A_adapter(ctr_final_states).reshape(-1, self.r, self.input_dim).permute(0, 2, 1)
            lora_B_weights = self.lora_B_adapter(ctr_final_states).reshape(-1, self.r, self.output_dim)
            # lora_A_weights = torch.mean(self.lora_A_adapter.weight,dim=-1).reshape(self.r,self.input_dim).permute(1,0).unsqueeze(0).repeat(x.shape[0],1,1)
            # lora_B_weights = torch.mean(self.lora_B_adapter.weight,dim=-1).reshape(self.output_dim,self.r).permute(1,0).unsqueeze(0).repeat(x.shape[0],1,1)

            return torch.bmm(torch.bmm(self.lora_dropout(x), lora_A_weights), lora_B_weights) * self.scaling



# class LoreclayerV(nn.Module):
#     def __init__(self, lora_config, ctr_hidden_size, ctr_final_dim, hidden_dim=5120):
#         super().__init__()
#         self.r = lora_config.r
#         self.hidden_dim = hidden_dim
#         self.alpha = lora_config.lora_alpha
#         self.ctr_final_dim = ctr_final_dim

#         self.ctr_hidden_states = None
#         self.ctr_final_layer = nn.Sequential(
#             nn.Linear(ctr_hidden_size, self.ctr_final_dim),
#             nn.ReLU(),
#             nn.LayerNorm(self.ctr_final_dim)
#         )
        
#         self.lora_A_adapter = nn.Linear(self.ctr_final_dim, self.r, bias=False) 
#         self.lora_B_adapter = nn.Linear(self.ctr_final_dim, self.hidden_dim, bias=False) 
#         self.lora_dropout = nn.Dropout(p=lora_config.lora_dropout)
#         self.scaling = self.alpha / self.r

#     def reset_lorec_parameters(self):
#         # initialize A the same way as the default for nn.Linear and B to zero
#         nn.init.kaiming_uniform_(self.lora_A_adapter.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B_adapter.weight)

#     def forward(self, x):
#         ctr_final_states = self.ctr_final_layer(self.ctr_hidden_states)
#         lora_A_weights = self.lora_A_adapter(ctr_final_states)
#         lora_B_weights = self.lora_B_adapter(ctr_final_states)
        
#         return torch.matmul(torch.matmul(self.lora_dropout(x), A) * lora_A_weights.unsqueeze(1), B) * lora_B_weights.unsqueeze(1) * self.scaling

#         # return torch.bmm(torch.bmm(self.lora_dropout(x),self.lora_A_weights),self.lora_B_weights)*self.scaling


class Loreclayer_ft(nn.Module):
    def __init__(self, lora_config, ctr_hidden_size, ctr_final_dim, input_dim=5120, output_dim=5120, initialization_method = 'kaiming'):
        super().__init__()
        self.lora_config = lora_config
        self.r = lora_config.r
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = lora_config.lora_alpha
        self.ctr_final_dim = ctr_final_dim
        self.initialization_method = initialization_method
        
        self.ctr_hidden_states = None
        self.ctr_final_layer = nn.Sequential(
            nn.LayerNorm(ctr_hidden_size),
            nn.Linear(ctr_hidden_size, self.ctr_final_dim),
        )
        enable_bias = self.lora_config.all_init_one
        self.lora_A_adapter = nn.Linear(self.ctr_final_dim, self.r, bias=enable_bias) 
        self.lora_B_adapter = nn.Linear(self.ctr_final_dim, self.output_dim, bias=enable_bias) 
        self.lora_dropout = nn.Dropout(p=lora_config.lora_dropout)
        self.lora_A = nn.Linear(self.input_dim, lora_config.r, bias=False)
        self.lora_B = nn.Linear(lora_config.r, self.output_dim, bias=False)
        self.scaling = self.alpha / self.r

    def reset_lorec_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        
        # init self.lora_A/B
        if self.initialization_method == 'kaiming':
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        else:
            nn.init.xavier_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # init self.lora_A/B_adapter
        if self.lora_config.all_init_one:
            nn.init.ones_(self.lora_A_adapter.bias)
            nn.init.ones_(self.lora_B_adapter.bias)
            nn.init.zeros_(self.lora_A_adapter.weight)
            nn.init.zeros_(self.lora_B_adapter.weight)
        else:
            if self.initialization_method == 'kaiming':
                nn.init.kaiming_uniform_(self.lora_A_adapter.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lora_B_adapter.weight, a=math.sqrt(5))
            else:
                nn.init.xavier_uniform_(self.lora_A_adapter.weight)
                nn.init.xavier_uniform_(self.lora_B_adapter.weight)
    
    def make_A_B_untrainable(self):
        self.lora_A.weight.requires_grad = False
        self.lora_B.weight.requires_grad = False

    def forward(self, x):
        ctr_final_states = self.ctr_final_layer(self.ctr_hidden_states)
        lora_A_scalar = self.lora_A_adapter(ctr_final_states)
        lora_B_scalar = self.lora_B_adapter(ctr_final_states)
        
        return self.lora_B(self.lora_A(self.lora_dropout(x)) * lora_A_scalar.unsqueeze(1)) * lora_B_scalar.unsqueeze(1) * self.scaling

        # return torch.bmm(torch.bmm(self.lora_dropout(x),self.lora_A_weights),self.lora_B_weights)*self.scaling


class Lorec(nn.Module):
    """
    Consist of one Language Model and a ctr-based lora generator.

    Args:
        config: LlamaConfig, CtrConfig
    """
    def __init__(self, llama_config, lora_config, ctr_config, mode='origin',head='ctr2', head_fix = False, init_method = 'kaiming',temperature=1,which_in_gate = 'None', USE_GC = False):
        super().__init__()
        self.LM = LlamaForCausalLM.from_pretrained(
            llama_config["model_path"],
            load_in_8bit=llama_config["load_in_8bit"],
            device_map=llama_config["device_map"],
        )
        print(temperature)
        self.layer_num = llama_config['layer_num']
        self.llama_hidden_dim = llama_config['hidden_dim']
        self.llama_intermediate_dim = llama_config['intermediate_dim']
        self.mode = mode
        self.r = lora_config.r
        self.ctr_final_dim = ctr_config.final_dim
        if ctr_config.out_layer == 'final':
            ctr_out_size = ctr_config.hidden_size
        elif ctr_config.out_layer == 'hidden':
            ctr_out_size = ctr_config.embed_size * (ctr_config.num_fields - 1)
        elif ctr_config.out_layer == 'user_reps':
            ctr_out_size = ctr_config.embed_size * (ctr_config.user_fields + 1)
        self.head = head
        if mode == 'origin':
            self.lorec_layers = nn.ModuleDict({})
            self.lorec_layers['q'] = nn.ModuleList([Loralayer(lora_config, self.llama_hidden_dim,self.llama_hidden_dim) for i in range(self.layer_num)])
            self.lorec_layers['v'] = nn.ModuleList([Loralayer(lora_config, self.llama_hidden_dim,self.llama_hidden_dim) for i in range(self.layer_num)])
            
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].reset_lora_parameters()
        elif mode == 'ctr':
            self.lorec_layers = nn.ModuleDict({})
            if 'q' in lora_config.target_modules:
                self.lorec_layers['q'] = nn.ModuleList([Loreclayer(lora_config, ctr_out_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method,temperature,which_in_gate) for i in range(self.layer_num)])
            if 'k' in lora_config.target_modules:
                self.lorec_layers['k'] = nn.ModuleList([Loreclayer(lora_config, ctr_out_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method,temperature,which_in_gate) for i in range(self.layer_num)])
            if 'v' in lora_config.target_modules:
                self.lorec_layers['v'] = nn.ModuleList([Loreclayer(lora_config, ctr_out_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method,temperature,which_in_gate) for i in range(self.layer_num)])
            if 'up' in lora_config.target_modules:
                self.lorec_layers['up'] = nn.ModuleList([Loreclayer(lora_config, ctr_out_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_intermediate_dim, init_method,temperature,which_in_gate) for i in range(self.layer_num)])
            if 'gate' in lora_config.target_modules:
                self.lorec_layers['gate'] = nn.ModuleList([Loreclayer(lora_config, ctr_out_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_intermediate_dim, init_method,temperature,which_in_gate) for i in range(self.layer_num)])
            if 'down' in lora_config.target_modules:
                self.lorec_layers['down'] = nn.ModuleList([Loreclayer(lora_config, ctr_out_size, self.ctr_final_dim, self.llama_intermediate_dim, self.llama_hidden_dim, init_method,temperature,which_in_gate) for i in range(self.layer_num)])
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].reset_lorec_parameters()
        
        # elif mode == 'vera':
        #     global A
        #     global B
        #     A = torch.randn(self.llama_hidden_dim, lora_config.r).to('cuda')
        #     B = torch.randn(lora_config.r, self.llama_hidden_dim).to('cuda')
        #     nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        #     nn.init.kaiming_uniform_(B, a=math.sqrt(5))
        #     self.lorec_layer_q = nn.ModuleList([LoreclayerV(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim) for i in range(self.layer_num)])
        #     self.lorec_layer_v = nn.ModuleList([LoreclayerV(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim) for i in range(self.layer_num)])
        #     for i in range(self.layer_num):
        #         self.lorec_layer_q[i].reset_lorec_parameters()
        #         self.lorec_layer_v[i].reset_lorec_parameters()
        elif mode == 'finetune':
            
            self.lorec_layers = nn.ModuleDict({})
            if 'q' in lora_config.target_modules:
                self.lorec_layers['q'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            if 'k' in lora_config.target_modules:
                self.lorec_layers['k'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            if 'v' in lora_config.target_modules:
                self.lorec_layers['v'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            if 'up' in lora_config.target_modules:
                self.lorec_layers['up'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_intermediate_dim, init_method) for i in range(self.layer_num)])
            if 'gate' in lora_config.target_modules:
                self.lorec_layers['gate'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_intermediate_dim, init_method) for i in range(self.layer_num)])
            if 'down' in lora_config.target_modules:
                self.lorec_layers['down'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_intermediate_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].reset_lorec_parameters()
            if lora_config.rella_model_pt:
                with torch.no_grad():
                    rella_model_dict = torch.load(lora_config.rella_model_pt)
                    for i in range(self.layer_num):
                        self.lorec_layers['q'][i].lora_A.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight']
                        self.lorec_layers['q'][i].lora_B.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight']
                        self.lorec_layers['v'][i].lora_A.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight']
                        self.lorec_layers['v'][i].lora_B.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight']
                for i in range(self.layer_num):
                    self.lorec_layers['q'][i].make_A_B_untrainable()
                    self.lorec_layers['v'][i].make_A_B_untrainable()

        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        # ctr head and model 
        
        if mode in ['ctr', 'vera', 'finetune']:
            if head == 'lm':
                print('it is lm_head')
            elif head == 'ctr2':
                if head_fix == False:
                    self.LM.ctr_head = nn.Linear(llama_config['hidden_dim'], 2)
                    self.LM.ctr_head.weight.data = self.LM.lm_head.weight.data[[1939, 3869], :]
                    nn.init.zeros_(self.LM.ctr_head.bias)
                else:
                    self.LM.ctr_head = nn.Linear(llama_config['hidden_dim'], 2,bias=False)
                    self.LM.ctr_head.weight.data = self.LM.lm_head.weight.data[[1939, 3869], :]
                del self.LM.lm_head
            elif head == 'ctr1':
                self.LM.ctr_head = nn.Linear(llama_config['hidden_dim'], 1)
                del self.LM.lm_head
            else:
                raise NotImplementedError(f"Unsupported head: {mode}")
        
            self.CTR_model = BaseModel.from_config(ctr_config)
            self.CTR_model.apply(weight_init)
            for name, param in self.CTR_model.named_parameters():
                param.requires_grad = False

        if llama_config["load_in_8bit"] is True :
            self.LM = prepare_model_for_int8_training(self.LM, use_gradient_checkpointing=USE_GC)

        for name, param in self.LM.named_parameters():
            param.requires_grad = True if "ctr_head" in name else False

        if head in ['ctr1','ctr2'] and head_fix :
            self.LM.ctr_head.weight.requires_grad = False

        self.handles = []
        for i in range(self.layer_num): 
            # partial is important and lambda is a piece of shit
            if 'q' in lora_config.target_modules:
                handle_q = self.LM.model.layers[i].self_attn.q_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['q'][i])) 
            if 'k' in lora_config.target_modules:
                handle_k = self.LM.model.layers[i].self_attn.k_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['k'][i]))
            if 'v' in lora_config.target_modules:
                handle_v = self.LM.model.layers[i].self_attn.v_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['v'][i]))
            if 'up' in lora_config.target_modules:
                handle_u = self.LM.model.layers[i].mlp.up_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['up'][i]))
            if 'gate' in lora_config.target_modules:
                handle_g = self.LM.model.layers[i].mlp.gate_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['gate'][i])) 
            if 'down' in lora_config.target_modules:
                handle_d = self.LM.model.layers[i].mlp.down_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['down'][i]))
        
        self.origin_state_dict = self.state_dict
        def new_state_dict():
            state_dict = self.origin_state_dict()
            to_return = {k: state_dict[k] for k in state_dict if "lorec_" in k}
            # print([(k, v.shape) for k, v in to_return.items()])
            return to_return
        self.state_dict = new_state_dict
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        X: Optional[torch.LongTensor] = None, 
        Y: Optional[torch.LongTensor] = None, 
        hist_ids: Optional[torch.LongTensor] = None, 
        hist_ratings: Optional[torch.LongTensor] = None, 
        hist_mask: Optional[torch.LongTensor] = None, 
    ):
        if self.mode == 'origin':
            return self.LM(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        elif self.mode in ['ctr', 'vera', 'finetune']:
            # batch_size = X.shape[0]
            ctr_hidden_states = self.CTR_model(X, Y, hist_ids, hist_ratings, hist_mask).detach()
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].ctr_hidden_states = ctr_hidden_states
            
            # not changing head
            if self.head == 'lm':
                return self.LM(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
            )
            # changing to ctr head
            outputs = self.LM.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]
            assert (labels[:, -1] == 29889).sum() == Y.shape[0]
            assert (labels[:, -2] == 1939).sum() + (labels[:, -2] == 3869).sum() == Y.shape[0]
                
            if self.head == 'ctr2':
                logits = self.LM.ctr_head(hidden_states[:, -3, :])
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, Y)
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                )
            elif self.head == 'ctr1':
                logits = self.LM.ctr_head(hidden_states[:,-3,:])
                loss_fn = nn.BCELoss()
                logits = logits.squeeze()
                logits = torch.nn.Sigmoid()(logits)
                loss = loss_fn(logits, Y.squeeze().float())
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                )
            
            
class Lorec_mis(nn.Module):
    """
    Consist of one Language Model and a ctr-based lora generator.

    Args:
        config: Mis_Config, CtrConfig
    """
    def __init__(self, mis_config, lora_config, ctr_config, mode='origin',head='ctr2', head_fix = False, init_method = 'kaiming'):
        super().__init__()
        self.LM = AutoModelForCausalLM.from_pretrained(
            mis_config["model_path"],
            load_in_8bit=mis_config["load_in_8bit"],
            device_map=mis_config["device_map"],
        )
        self.layer_num = mis_config['layer_num'] #32
        self.mis_hidden_dim1 = mis_config['hidden_dim1']
        self.mis_hidden_dim2 = mis_config['hidden_dim2']
        self.mis_intermediate_dim = mis_config['intermediate_dim']
        self.mode = mode
        self.r = lora_config.r
        self.ctr_final_dim = ctr_config.final_dim
        self.head = head
        if mode == 'origin':
            self.lorec_layers = nn.ModuleDict({})
            self.lorec_layers['q'] = nn.ModuleList([Loralayer(lora_config, self.mis_hidden_dim1,self.mis_hidden_dim1) for i in range(self.layer_num)])
            self.lorec_layers['k'] = nn.ModuleList([Loralayer(lora_config, self.mis_hidden_dim1,self.mis_hidden_dim2) for i in range(self.layer_num)])
            self.lorec_layers['v'] = nn.ModuleList([Loralayer(lora_config, self.mis_hidden_dim1,self.mis_hidden_dim2) for i in range(self.layer_num)])
            self.lorec_layers['o'] = nn.ModuleList([Loralayer(lora_config, self.mis_hidden_dim1,self.mis_hidden_dim1) for i in range(self.layer_num)])
            
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].reset_lora_parameters()
        elif mode == 'ctr':
            self.lorec_layers = nn.ModuleDict({})
            if 'q' in lora_config.target_modules:
                self.lorec_layers['q'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_hidden_dim1, self.mis_hidden_dim1, init_method) for i in range(self.layer_num)])
            if 'k' in lora_config.target_modules:
                self.lorec_layers['k'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_hidden_dim1, self.mis_hidden_dim2, init_method) for i in range(self.layer_num)])
            if 'v' in lora_config.target_modules:
                self.lorec_layers['v'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_hidden_dim1, self.mis_hidden_dim2, init_method) for i in range(self.layer_num)])
            if 'v' in lora_config.target_modules:
                self.lorec_layers['o'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_hidden_dim1, self.mis_hidden_dim1, init_method) for i in range(self.layer_num)])
            if 'up' in lora_config.target_modules:
                self.lorec_layers['up'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_hidden_dim1, self.mis_intermediate_dim, init_method) for i in range(self.layer_num)])
            if 'gate' in lora_config.target_modules:
                self.lorec_layers['gate'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_hidden_dim1, self.mis_intermediate_dim, init_method) for i in range(self.layer_num)])
            if 'down' in lora_config.target_modules:
                self.lorec_layers['down'] = nn.ModuleList([Loreclayer(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.mis_intermediate_dim, self.mis_hidden_dim1, init_method) for i in range(self.layer_num)])
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].reset_lorec_parameters()
         
        elif mode == 'finetune':
            
            self.lorec_layers = nn.ModuleDict({})
            if 'q' in lora_config.target_modules:
                self.lorec_layers['q'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            if 'k' in lora_config.target_modules:
                self.lorec_layers['k'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            if 'v' in lora_config.target_modules:
                self.lorec_layers['v'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            if 'up' in lora_config.target_modules:
                self.lorec_layers['up'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_intermediate_dim, init_method) for i in range(self.layer_num)])
            if 'gate' in lora_config.target_modules:
                self.lorec_layers['gate'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_hidden_dim, self.llama_intermediate_dim, init_method) for i in range(self.layer_num)])
            if 'down' in lora_config.target_modules:
                self.lorec_layers['down'] = nn.ModuleList([Loreclayer_ft(lora_config, ctr_config.hidden_size, self.ctr_final_dim, self.llama_intermediate_dim, self.llama_hidden_dim, init_method) for i in range(self.layer_num)])
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].reset_lorec_parameters()
            if lora_config.rella_model_pt:
                with torch.no_grad():
                    rella_model_dict = torch.load(lora_config.rella_model_pt)
                    for i in range(self.layer_num):
                        self.lorec_layers['q'][i].lora_A.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight']
                        self.lorec_layers['q'][i].lora_B.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight']
                        self.lorec_layers['v'][i].lora_A.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight']
                        self.lorec_layers['v'][i].lora_B.weight.data = rella_model_dict[f'base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight']
                for i in range(self.layer_num):
                    self.lorec_layers['q'][i].make_A_B_untrainable()
                    self.lorec_layers['v'][i].make_A_B_untrainable()

        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        # ctr head and model 
        
        if mode in ['ctr', 'vera', 'finetune']:
            if head == 'lm':
                print('it is lm_head')
            elif head == 'ctr2':
                if head_fix == False:
                    self.LM.ctr_head = nn.Linear(mis_config['hidden_dim1'], 2)
                    self.LM.ctr_head.weight.data = self.LM.lm_head.weight.data[[2501, 5613], :]
                    nn.init.zeros_(self.LM.ctr_head.bias)
                else:
                    self.LM.ctr_head = nn.Linear(mis_config['hidden_dim1'], 2,bias=False)
                    self.LM.ctr_head.weight.data = self.LM.lm_head.weight.data[[2501, 5613], :]
                del self.LM.lm_head
            elif head == 'ctr1':
                self.LM.ctr_head = nn.Linear(mis_config['hidden_dim'], 1)
                del self.LM.lm_head
            else:
                raise NotImplementedError(f"Unsupported head: {mode}")
        
            self.CTR_model = BaseModel.from_config(ctr_config)
            self.CTR_model.apply(weight_init)
            for name, param in self.CTR_model.named_parameters():
                param.requires_grad = False

        if mis_config["load_in_8bit"] is True:
            self.LM = prepare_model_for_int8_training(self.LM)
        
        for name, param in self.LM.named_parameters():
            param.requires_grad = True if "ctr_head" in name else False

        if head in ['ctr1','ctr2'] and head_fix :
            self.LM.ctr_head.weight.requires_grad = False

        self.handles = []
        for i in range(self.layer_num): 
            # partial is important and lambda is a piece of shit
            if 'q' in lora_config.target_modules:
                handle_q = self.LM.model.layers[i].self_attn.q_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['q'][i])) 
            if 'k' in lora_config.target_modules:
                handle_k = self.LM.model.layers[i].self_attn.k_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['k'][i]))
            if 'v' in lora_config.target_modules:
                handle_v = self.LM.model.layers[i].self_attn.v_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['v'][i]))
            if 'o' in lora_config.target_modules:
                handle_v = self.LM.model.layers[i].self_attn.o_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['o'][i]))
            if 'up' in lora_config.target_modules:
                handle_u = self.LM.model.layers[i].mlp.up_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['up'][i]))
            if 'gate' in lora_config.target_modules:
                handle_g = self.LM.model.layers[i].mlp.gate_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['gate'][i])) 
            if 'down' in lora_config.target_modules:
                handle_d = self.LM.model.layers[i].mlp.down_proj.register_forward_hook(partial(lorec_hook,self.lorec_layers['down'][i]))
        
        self.origin_state_dict = self.state_dict
        def new_state_dict():
            state_dict = self.origin_state_dict()
            to_return = {k: state_dict[k] for k in state_dict if "lorec_" in k}
            # print([(k, v.shape) for k, v in to_return.items()])
            return to_return
        self.state_dict = new_state_dict
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None, 
        X: Optional[torch.LongTensor] = None, 
        Y: Optional[torch.LongTensor] = None, 
        hist_ids: Optional[torch.LongTensor] = None, 
        hist_ratings: Optional[torch.LongTensor] = None, 
        hist_mask: Optional[torch.LongTensor] = None, 
    ):
        if self.mode == 'origin':
            return self.LM(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        elif self.mode in ['ctr', 'vera', 'finetune']:
            # batch_size = X.shape[0]
            ctr_hidden_states = self.CTR_model(X, Y, hist_ids, hist_ratings, hist_mask).detach()
            for key, layer in self.lorec_layers.items():
                for i in range(self.layer_num):
                    self.lorec_layers[key][i].ctr_hidden_states = ctr_hidden_states
            
            # not changing head
            if self.head == 'lm':
                return self.LM(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
            )
            # changing to ctr head
            outputs = self.LM.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]
            assert (labels[:, -1] == 29889).sum() == Y.shape[0]
            assert (labels[:, -2] == 1939).sum() + (labels[:, -2] == 3869).sum() == Y.shape[0]
                
            if self.head == 'ctr2':
                logits = self.LM.ctr_head(hidden_states[:, -3, :])
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, Y)
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                )
            elif self.head == 'ctr1':
                logits = self.LM.ctr_head(hidden_states[:,-3,:])
                loss_fn = nn.BCELoss()
                logits = logits.squeeze()
                logits = torch.nn.Sigmoid()(logits)
                loss = loss_fn(logits, Y.squeeze().float())
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                )