import numpy as np
import torch.nn as nn
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
         nn.init.xavier_normal_(m.weight)       

def lora_tsne(model,llama_config):
    for i in range(llama_config['layer_num']):
        a = model.lorec_layers['q'][i].lora_A_adapter.weight.permute(1,0)
        tsne = TSNE(n_components = 2, random_state = 0, perplexity=10)
        a = a.detach().numpy()
        data_2d = tsne.fit_transform(a)
        data_2d = tsne.fit_transform(a)
        plt.figure(figsize=(8,6))
        plt.scatter(data_2d[:,0], data_2d[:,1])
        plt.savefig(f'q-{i}.png')
    for i in range(llama_config['layer_num']):
        a = model.lorec_layers['v'][i].lora_A_adapter.weight.permute(1,0)
        tsne = TSNE(n_components = 2, random_state = 0, perplexity=10)
        a = a.detach().numpy()
        data_2d = tsne.fit_transform(a)
        data_2d = tsne.fit_transform(a)
        plt.figure(figsize=(8,6))
        plt.scatter(data_2d[:,0], data_2d[:,1])
        plt.savefig(f'v-{i}.png')

def a():
    print(1)

def get_ctr_config(args):
    if args.dataset == "GoodReads":
        if args.ctr_K == 60:
            ctr_config = {
                "item_field_idx": 1,
                "embed_size": 32,
                "hidden_size": 256, 
                "num_hidden_layers": 3, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 1,
                "output_dim": 1,
                "num_fields": 17, #15+1+1(id15+hist_id1+rating1)
                "num_features": 5787895, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
        if args.ctr_K == 30:
            ctr_config = {
                "item_field_idx": 1,
                "embed_size": 64,
                "hidden_size": 256, 
                "num_hidden_layers": 3, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 1,
                "output_dim": 1,
                "num_fields": 17, #8+1+1(id8+hist_id1+rating1)
                "num_features": 5787895, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
        if args.ctr_K == 10:
            ctr_config = {
                "item_field_idx": 1,
                "embed_size": 64,
                "hidden_size": 256, 
                "num_hidden_layers": 3, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 1,
                "output_dim": 1,
                "num_fields": 17, #8+1+1(id8+hist_id1+rating1)
                "num_features": 5787895, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
        
    elif args.dataset == "BookCrossing":
        if args.ctr_K == 60:
            ctr_config = {
                "item_field_idx": 3,
                "embed_size": 8,
                "hidden_size": 32, 
                "num_hidden_layers": 3, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 1,
                "output_dim": 1,
                "num_fields": 10, #8+1+1(id8+hist_id1+rating1)
                "num_features": 912279, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
        if args.ctr_K == 30:
            ctr_config = {
                "item_field_idx": 3,
                "embed_size": 32,
                "hidden_size": 64, 
                "num_hidden_layers": 3, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 2,
                "output_dim": 1,
                "num_fields": 10, #8+1+1(id8+hist_id1+rating1)
                "num_features": 912279, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
    elif args.dataset == 'ml-25m':
        if args.ctr_K == 30:
            ctr_config = {
                "item_field_idx": 1,
                "embed_size": 64,
                "hidden_size": 256, 
                "num_hidden_layers": 3, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 2,
                "output_dim": 1,
                "num_fields": 6, #4+1+1(id4+hist_id1+rating1)
                "user_fields": 1,
                "num_features": 280576, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
        elif args.ctr_K ==10:
            ctr_config = {
                "item_field_idx": 1,
                "embed_size": 32,
                "hidden_size": 256, 
                "num_hidden_layers": 1, 
                "hidden_dropout_rate": args.ctr_dropout, 
                "hidden_act": 'relu',
                "num_attn_layers": 2,
                "output_dim": 1,
                "num_fields": 6, #4+1+1(id4+hist_id1+rating1)
                "user_fields": 1,
                "num_features": 280576, 
                'enable_hist_embed': True, 
                "enable_rating_embed": True, 
                'embed_dropout_rate': 0.0, 
                'model_name': 'din', 
                "final_dim": args.lora_assemble_num,
                "out_layer": args.ctr_out_layer,
            }
        elif args.ctr_K ==60:
            if args.ret:
                ctr_config = {
                    "item_field_idx": 1,
                    "embed_size": 64,
                    "hidden_size": 256, 
                    "num_hidden_layers": 3, 
                    "hidden_dropout_rate": args.ctr_dropout, 
                    "hidden_act": 'relu',
                    "num_attn_layers": 3,
                    "output_dim": 1,
                    "num_fields": 6, #4+1+1(id4+hist_id1+rating1)
                    "user_fields": 1,
                    "num_features": 280576, 
                    'enable_hist_embed': True, 
                    "enable_rating_embed": True, 
                    'embed_dropout_rate': 0.0, 
                    'model_name': 'din', 
                    "final_dim": args.lora_assemble_num,
                    "out_layer": args.ctr_out_layer,
                }
            else:
                ctr_config = {
                    "item_field_idx": 1,
                    "embed_size": 32,
                    "hidden_size": 256, 
                    "num_hidden_layers": 3, 
                    "hidden_dropout_rate": args.ctr_dropout, 
                    "hidden_act": 'relu',
                    "num_attn_layers": 3,
                    "output_dim": 1,
                    "num_fields": 6, #4+1+1(id4+hist_id1+rating1)
                    "user_fields": 1,
                    "num_features": 280576, 
                    'enable_hist_embed': True, 
                    "enable_rating_embed": True, 
                    'embed_dropout_rate': 0.0, 
                    'model_name': 'din', 
                    "final_dim": args.lora_assemble_num,
                    "out_layer": args.ctr_out_layer,
                }
    elif args.dataset == 'ml-1m':
        ctr_config = {
            "item_field_idx": 5,
            "embed_size": 64,
            "hidden_size": 256, 
            "num_hidden_layers": 3, 
            "hidden_dropout_rate": args.ctr_dropout, 
            "hidden_act": 'relu',
            "num_attn_layers": 1,
            "output_dim": 1,
            "num_fields": 10, #8+1+1(id4+hist_id1+rating1)
            "num_features": 16944, 
            'enable_hist_embed': True, 
            "enable_rating_embed": True, 
            'embed_dropout_rate': 0.0, 
            'model_name': 'din', 
            "final_dim": args.lora_assemble_num,
            "out_layer": args.ctr_out_layer,
        }
    return ctr_config