import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import logging
from typing import Dict, Optional, Tuple
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import math

from ctr_base.layers import Embeddings, InnerProductLayer, OuterProductLayer, MLPBlock, get_act, \
    ProductLayer, CrossNetV2, FGCNNBlock, SqueezeExtractionLayer, \
    BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, SelfAttention, \
    CIN, MultiHeadSelfAttention, DIN_Attention, MultiHeadTargetAttention
from ctr_base.config import Config


logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    used_params = []

    def __init__(self, model_name="BaseModel", config: Config=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.config = config

    @classmethod
    def from_config(cls, config: Config):
        model_name_lower = config.model_name.lower()
        if model_name_lower == "lr":
            model_class = LR
        elif model_name_lower == "fm":
            model_class = FM
        elif model_name_lower == "dnn":
            model_class = DNN
        elif model_name_lower == "trans":
            model_class = Transformer
        elif model_name_lower == "deepfm":
            model_class = DeepFM
        elif model_name_lower == "xdeepfm":
            model_class = xDeepFM
        elif model_name_lower == "dcnv2":
            model_class = DCNV2
        elif model_name_lower == "fgcnn":
            model_class = FGCNN
        elif model_name_lower == "fibinet":
            model_class = FiBiNet
        elif model_name_lower == "fignn":
            model_class = FiGNN
        elif model_name_lower == "autoint":
            model_class = AutoInt
        elif model_name_lower == "din":
            model_class = DIN
        elif model_name_lower == "gru4rec":
            model_class = GRU4Rec
        elif model_name_lower == "sasrec":
            model_class = SASRec
        elif model_name_lower == "caser":
            model_class = Caser
        elif model_name_lower == "eta":
            model_class = ETA
        else:
            raise NotImplementedError
        model = model_class(config)
        return model

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logger.info(f"total number of parameters: {total_params}")

    def validate_model_config(self):
        if self.model_name.lower() in ["lr", "fm", "dcnv2", ]:
            assert self.config.output_dim == 1, f"model {self.model_name} requires output_dim == 1"

        if self.model_name.lower() in ["trans"]:
            assert self.config.embed_size == self.config.hidden_size, \
                f"model {self.model_name} requires embed_size == hidden_size"

        logger.info(f"  model_name = {self.model_name}")
        for key in self.used_params:
            logger.info(f"  {key} = {getattr(self.config, key)}")

    def get_outputs(self, logits, labels=None):
        outputs = {
            "logits": logits,
        }
        if labels is not None:
            if self.config.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.config.output_dim)), labels.long())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())
            outputs["loss"] = loss

        return outputs
    
    def get_hist_embed(self, feat_embed_table, feat_embed, hist_info, hist_mask):
        hist_embed = feat_embed_table(hist_info)
        hist_embed = torch.sum(hist_embed * hist_mask.unsqueeze(-1), dim=1) / torch.sum(hist_mask, dim=1, keepdim=True)
        feat_embed = torch.cat([feat_embed, hist_embed.unsqueeze(1)], dim=1)
        return feat_embed
        

class LR(BaseModel):
    used_params = ["output_dim"]

    def __init__(self, config: Config):
        super().__init__(model_name="LR", config=config)
        self.embed_w = nn.Embedding(config.num_features, embedding_dim=1)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids, labels=None):
        batch_size = input_ids.shape[0]
        wx = self.embed_w(input_ids)
        logits = wx.sum(dim=1) + self.bias
        outputs = self.get_outputs(logits, labels)
        
        return outputs


class FM(BaseModel):
    used_params = ["embed_size", "output_dim"]

    def __init__(self, config: Config):
        super().__init__(model_name="FM", config=config)
        self.lr_layer = LR(config)
        self.embed = Embeddings(config)
        if config.pretrain:
            # self.ip_layer = InnerProductLayer(num_fields=config.num_fields, output="inner_product")
            # self.feat_encoder = nn.Linear(config.num_fields * (config.num_fields - 1) // 2 + config.num_fields + 1, config.proj_size)
            self.ip_layer = InnerProductLayer(num_fields=config.num_fields)
            self.feat_encoder = nn.Linear(1, config.proj_size)
            self.criterion = IndexLinear(config)
        else:
            self.ip_layer = InnerProductLayer(num_fields=config.num_fields)

    def forward(self, input_ids, labels=None, masked_index=None):
        feat_embed = self.embed(input_ids)
        
        if self.config.pretrain:
            # Pretrain phase
            lr_vec = self.lr_layer(input_ids)[0]
            fm_vec = self.ip_layer(feat_embed)
            # print(lr_vec.shape)
            # print(fm_vec.shape)
            enc_output = self.feat_encoder(torch.cat([lr_vec, fm_vec], dim=1))
            # print(enc_output.shape)
            # assert 0
            selected_output = enc_output.unsqueeze(1).repeat(1, masked_index.shape[1], 1)
            loss, G_logits, G_features = self.criterion(labels, selected_output)
            total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
            outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        else:
            lr_logits = self.lr_layer(input_ids)[0]
            dot_sum = self.ip_layer(feat_embed)
            logits = dot_sum + lr_logits
            outputs = self.get_outputs(logits, labels)

        return outputs


class DNN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "output_dim"]

    def __init__(self, config: Config):
        super(DNN, self).__init__(model_name="DNN", config=config)
        
        self.embed = Embeddings(config)
        self.dnn = MLPBlock(
            input_dim=config.embed_size * config.num_fields, 
            hidden_size=config.hidden_size, 
            num_hidden_layers=config.num_hidden_layers, 
            hidden_dropout_rate=config.hidden_dropout_rate, 
            hidden_act=config.hidden_act, 
        )
        self.fc_out = nn.Linear(config.hidden_size, config.output_dim)

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embedding
        feat_embed = self.embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
        feat_embed = feat_embed.flatten(start_dim=1)
        
        # Feature interaction & logits
        final_output = self.dnn(feat_embed)
        logits = self.fc_out(final_output)
        
        # Loss
        outputs = self.get_outputs(logits, labels)
        
        return outputs


class DeepFM(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", 
                   "hidden_act", "output_dim"]

    def __init__(self, config: Config):
        super(DeepFM, self).__init__(model_name="DeepFM", config=config)
        
        self.embed = Embeddings(config)  
        self.dnn = MLPBlock(
            input_dim=config.num_fields * config.embed_size, 
            hidden_size=config.hidden_size, 
            num_hidden_layers=config.num_hidden_layers, 
            hidden_dropout_rate=config.hidden_dropout_rate, 
            hidden_act=config.hidden_act
        )
        self.fc_out = nn.Linear(config.hidden_size, config.output_dim)
        self.ip_layer = InnerProductLayer(num_fields=config.num_fields)
        self.lr_layer = LR(config)

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embedding
        feat_embed = self.embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
        
        # Feature interaction & logits
        dnn_output = self.dnn(feat_embed.flatten(start_dim=1))
        logits = self.fc_out(dnn_output)
        logits += self.ip_layer(feat_embed)
        logits += self.lr_layer(input_ids)["logits"]
        
        # Loss
        outputs = self.get_outputs(logits, labels)
        
        return outputs


class xDeepFM(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "cin_layer_units", "use_lr", "output_dim"]

    def __init__(self, config: Config):
        super(xDeepFM, self).__init__(model_name="xDeepFM", config=config)

        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        cin_layer_units = [int(c) for c in config.cin_layer_units.split("|")]
        self.cin = CIN(config.num_fields, cin_layer_units)
        self.final_dim = sum(cin_layer_units)
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(
                input_dim=input_dim, 
                hidden_size=config.hidden_size, 
                num_hidden_layers=config.num_hidden_layers, 
                hidden_dropout_rate=config.hidden_dropout_rate, 
                hidden_act=config.hidden_act, 
            )
            self.final_dim += config.hidden_size
        self.fc = nn.Linear(self.final_dim, config.output_dim)
        self.lr_layer = LR(config) if config.use_lr else None

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embedding
        feat_embed = self.embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
        
        # Feature interaction & logits
        final_output = self.cin(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_output = self.dnn(feat_embed.flatten(start_dim=1))
            final_output = torch.cat([final_output, dnn_output], dim=1)
        logits = self.fc(final_output)
        if self.config.use_lr:
            logits += self.lr_layer(input_ids)["logits"]
        
        # Loss
        outputs = self.get_outputs(logits, labels)
        
        return outputs


class DCNV2(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "num_cross_layers", "output_dim"]

    def __init__(self, config: Config):
        super(DCNV2, self).__init__(model_name="DCNV2", config=config)
        
        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        self.cross_net = CrossNetV2(input_dim, config.num_cross_layers)
        self.final_dim = input_dim
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(
                input_dim=input_dim,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                hidden_dropout_rate=config.hidden_dropout_rate,
                hidden_act=config.hidden_act,
            )
            self.final_dim += config.hidden_size
        self.fc_out = nn.Linear(self.final_dim, config.output_dim)

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embedding
        feat_embed = self.embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
        feat_embed = feat_embed.flatten(start_dim=1)
        
        # Feature interaction & logits
        final_output = self.cross_net(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_output = self.dnn(feat_embed)
            final_output = torch.cat([final_output, dnn_output], dim=-1)
        logits = self.fc_out(final_output)
        
        # Loss
        outputs = self.get_outputs(logits, labels)

        return outputs


class FGCNN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "share_embedding", "channels", "kernel_heights", "pooling_sizes", "recombined_channels",
                   "conv_act", "output_dim"]

    def __init__(self, config: Config):
        super(FGCNN, self).__init__(model_name="fgcnn", config=config)
        
        self.embed = Embeddings(config)
        if not config.share_embedding:
            self.fg_embed = Embeddings(config)
        channels = [int(c) for c in config.channels.split("|")]
        kernel_heights = [int(c) for c in config.kernel_heights.split("|")]
        pooling_sizes = [int(c) for c in config.pooling_sizes.split("|")]
        recombined_channels = [int(c) for c in config.recombined_channels.split("|")]
        self.fgcnn_layer = FGCNNBlock(
            config.num_fields, 
            config.embed_size, 
            channels=channels, 
            kernel_heights=kernel_heights, 
            pooling_sizes=pooling_sizes, 
            recombined_channels=recombined_channels, 
            activation=config.conv_act, 
            batch_norm=True, 
        )
        final_dim, total_features = self.compute_input_dim(
            config.embed_size, 
            config.num_fields, 
            channels, 
            pooling_sizes, 
            recombined_channels,
        )
        self.ip_layer = InnerProductLayer(total_features, output="inner_product")
        
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(
                input_dim=final_dim, 
                hidden_size=config.hidden_size, 
                num_hidden_layers=config.num_hidden_layers, 
                hidden_dropout_rate=config.hidden_dropout_rate, 
                hidden_act=config.hidden_act
            )
            final_dim = config.hidden_size
        self.fc_out = nn.Linear(final_dim, 1)

    def compute_input_dim(
        self,  
        embedding_dim, 
        num_fields, 
        channels, 
        pooling_sizes, 
        recombined_channels
    ):
        total_features = num_fields
        input_height = num_fields
        for i in range(len(channels)):
            input_height = int(math.ceil(input_height / pooling_sizes[i]))
            total_features += input_height * recombined_channels[i]
        final_dim = int(total_features * (total_features - 1) / 2) \
                    + total_features * embedding_dim
        return final_dim, total_features

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embedding
        feat_embed = self.embed(input_ids)
        feat_embed2 = feat_embed if self.config.share_embedding else self.fg_embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
            feat_embed2 = self.get_hist_embed(self.embed, feat_embed2, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
            feat_embed2 = self.get_hist_embed(self.embed, feat_embed2, hist_ratings, hist_mask)
        
        # Feature interaction & logits
        conv_in = torch.unsqueeze(feat_embed2, 1)
        new_feat_embed = self.fgcnn_layer(conv_in)
        combined_feat_embed = torch.cat([feat_embed, new_feat_embed], dim=1)
        ip_vec = self.ip_layer(combined_feat_embed)
        final_output = torch.cat([combined_feat_embed.flatten(start_dim=1), ip_vec], dim=1)
        if self.config.num_hidden_layers > 0:
            final_output = self.dnn(final_output)
        logits = self.fc_out(final_output)
        
        # Loss
        outputs = self.get_outputs(logits, labels)

        return outputs


class FiBiNet(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "use_lr", "reduction_ratio", "bilinear_type", "output_dim"]

    def __init__(self, config: Config):
        super(FiBiNet, self).__init__(model_name="FiBiNet", config=config)
        
        self.embed = Embeddings(config)
        self.senet_layer = SqueezeExtractionLayer(config)
        self.bilinear_layer = BilinearInteractionLayer(config)
        self.lr_layer = LR(config) if config.use_lr else None
        self.final_dim = config.num_fields * (config.num_fields - 1) * config.embed_size
    
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(
                input_dim=self.final_dim,
                hidden_size=config.hidden_size, 
                num_hidden_layers=config.num_hidden_layers, 
                hidden_dropout_rate=config.hidden_dropout_rate, 
                hidden_act=config.hidden_act, 
            )
            self.final_dim = config.hidden_size
        self.fc_out = nn.Linear(self.final_dim, 1)

    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        
        # Embedding
        feat_embed = self.embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
        
        # Feature interaction & logits
        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)
        final_output = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        if self.config.num_hidden_layers > 0:
            final_output = self.dnn(final_output)
        logits = self.fc_out(final_output)
        if self.config.use_lr:
            logits += self.lr_layer(input_ids)["logits"]
        
        # Loss
        outputs = self.get_outputs(logits, labels)
        
        return outputs


class FiGNN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "res_conn", "reuse_graph_layer", "output_dim"]

    def __init__(self, config: Config):
        super(FiGNN, self).__init__(model_name="FiGNN", config=config)
        logger.warning("this model requires embed_size == hidden_size, only uses embed_size")
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == "RFD" and config.is_generator:
            self.embed = None
        else:
            self.embed = Embeddings(config)
        self.fignn = FiGNNBlock(config)

        if config.pretrain:
            final_dim = config.num_fields * config.embed_size
            self.create_pretraining_predictor(final_dim)
        else:
            self.fc = AttentionalPrediction(config)

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        h_out = self.fignn(feat_embed)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(h_out.flatten(start_dim=1), labels, masked_index)
        else:
            logits = self.fc(h_out)
            outputs = self.get_outputs(logits, labels)
        
        return outputs


class AutoInt(BaseModel):
    used_params = ["embed_size", "num_attn_layers", "attn_size", "num_attn_heads", 
                   "attn_probs_dropout_rate", "use_lr", "res_conn", "attn_scale", "output_dim", 
                   "dnn_size", "num_dnn_layers", "dnn_act", "dnn_drop"]

    def __init__(self, config: Config):
        super(AutoInt, self).__init__(model_name="AutoInt", config=config)

        self.embed = Embeddings(config)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(config.embed_size if i == 0 else config.num_attn_heads * config.attn_size,
                                     attention_dim=config.attn_size, 
                                     num_heads=config.num_attn_heads, 
                                     dropout_rate=config.attn_probs_dropout_rate, 
                                     use_residual=config.res_conn, 
                                     use_scale=config.attn_scale, 
                                     layer_norm=False,
                                     align_to="output") 
             for i in range(config.num_attn_layers)])
        final_dim = config.num_fields * config.attn_size * config.num_attn_heads

        self.attn_out = nn.Linear(final_dim, 1)
        self.lr_layer = LR(config) if config.use_lr else None
        self.dnn = MLPBlock(input_dim=final_dim,
                            hidden_size=config.dnn_size,
                            num_hidden_layers=config.num_dnn_layers,
                            hidden_dropout_rate=config.dnn_drop,
                            hidden_act=config.dnn_act) if config.num_dnn_layers else None
        self.dnn_out = nn.Linear(config.dnn_size, 1) if config.num_dnn_layers else None
            
    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        batch_size = input_ids.shape[0]
        feat_embed = self.embed(input_ids)
        if self.config.enable_hist_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ids, hist_mask)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)


        attention_out = self.self_attention(feat_embed)
        attention_out = torch.flatten(attention_out, start_dim=1)

        logits = self.attn_out(attention_out)
        if self.lr_layer is not None:
            logits += self.lr_layer(input_ids)["logits"]
        if self.dnn is not None:
            logits += self.dnn_out(self.dnn(feat_embed.flatten(start_dim=1)))
        outputs = self.get_outputs(logits, labels)
        return outputs


class Transformer(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "num_attn_heads", "intermediate_size", "output_reduction", 
                   "norm_first", "layer_norm_eps", "use_lr", "output_dim", 
                   "dnn_size", "num_dnn_layers", "dnn_act", "dnn_drop"]

    def __init__(self, config: Config):
        super(Transformer, self).__init__(model_name="trans", config=config)
        logger.warning("this model requires embed_size == hidden_size, only uses embed_size")
        
        # No need for the embedding layer for the generator of RFD
        if config.pretrain and config.pt_type == "RFD" and config.is_generator:
            self.embed = None
        else:
            self.embed = Embeddings(config)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attn_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_rate,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        if config.pretrain:
            final_dim = config.num_fields * config.embed_size
            self.create_pretraining_predictor(final_dim)
        else:
            if self.config.output_reduction == "fc":
                self.trans_out = nn.Linear(config.num_fields * config.embed_size, config.output_dim)
            elif self.config.output_reduction == "mean,fc" or self.config.output_reduction == "sum,fc":
                self.trans_out = nn.Linear(config.embed_size, config.output_dim)
            elif self.config.output_reduction == "attn,fc":
                self.field_reduction_attn = nn.Sequential(
                    nn.Linear(config.embed_size, config.embed_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(config.embed_size, 1),
                    nn.Softmax(dim=1),
                )
                self.trans_out = nn.Linear(config.embed_size, config.output_dim)
            else:
                raise NotImplementedError
            self.lr_layer = LR(config) if config.use_lr else None
            if config.num_dnn_layers > 0:
                self.mlp = MLPBlock(input_dim=config.num_fields * config.embed_size,
                                    hidden_size=config.dnn_size,
                                    num_hidden_layers=config.num_dnn_layers,
                                    hidden_dropout_rate=config.dnn_drop,
                                    hidden_act=config.dnn_act)
                self.mlp_out = nn.Linear(config.dnn_size, config.output_dim)
            else:
                self.mlp = None

    def forward(self, input_ids, labels=None, masked_index=None, feat_embed=None):
        batch_size = input_ids.shape[0]
        if feat_embed is None:
            feat_embed = self.embed(input_ids)
        enc_output = self.encoder(feat_embed)
        
        if self.config.pretrain:
            outputs = self.get_pretraining_output(enc_output.flatten(start_dim=1), labels, masked_index)
        else:
            # Finetune or train from scratch
            if self.config.output_reduction == "fc":
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == "mean,fc":
                enc_output = torch.sum(enc_output, dim=1) / self.config.num_fields
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == "sum,fc":
                enc_output = torch.sum(enc_output, dim=1)
                logits = self.trans_out(enc_output.flatten(start_dim=1))
            elif self.config.output_reduction == "attn,fc":
                attn_score = self.field_reduction_attn(enc_output)
                attn_feat = torch.sum(enc_output * attn_score, dim=1)
                logits = self.trans_out(attn_feat)
            if self.lr_layer is not None:
                logits += self.lr_layer(input_ids)[0]
            if self.mlp is not None:
                logits += self.mlp_out(self.mlp(feat_embed.flatten(start_dim=1)))
            outputs = self.get_outputs(logits, labels)
        return outputs


class DIN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "output_dim", "num_attn_layers"]

    def __init__(self, config: Config):
        super(DIN, self).__init__(model_name="DIN", config=config)
        
        self.item_field_idx = config.item_field_idx
        self.embed = Embeddings(config)
        hidden_act = config.hidden_act
        if isinstance(config.hidden_act, str) and config.hidden_act.lower() == "dice":
            hidden_act = [layers.Dice(config.hidden_size) for _ in range(config.num_hidden_layers)]
        
        self.final_dim = config.embed_size * config.num_fields
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(
                input_dim=config.embed_size * (config.num_fields - 1), 
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers, 
                hidden_dropout_rate=config.hidden_dropout_rate, 
                hidden_act=hidden_act, 
            )
            self.final_dim = config.hidden_size
        self.attention_layer = DIN_Attention(config)
        self.hist_mapping = nn.Sequential(
            nn.Linear(2*config.embed_size, config.embed_size),
            nn.ReLU()
        ) 
        self.fc_out = nn.Linear(self.final_dim, config.output_dim)


    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        """
        :param item_field_idx: index of item id field in all fields used, 5 for ml-1m, 3 for BookCrossing, 1 for ml-25m
        """
        batch_size = input_ids.shape[0]
        if self.config.out_layer == 'user_reps':
            return self.get_pretrained_user_reps(input_ids,labels,hist_ids,hist_ratings,hist_mask)
        # Embedding
        feat_embed = self.embed(input_ids)
        assert self.config.enable_hist_embed
        hist_embed = self.embed(hist_ids)
        if self.config.enable_rating_embed:
            rating_embed = self.embed(hist_ratings)
            hist_embed = torch.cat([hist_embed, rating_embed], dim=-1)
            hist_embed = self.hist_mapping(hist_embed)

        attention_output = self.attention_layer(feat_embed[:,self.item_field_idx,:], hist_embed, hist_mask)
        feat_embed = torch.cat([feat_embed, attention_output.unsqueeze(1)], dim=1)
        
        # Feature interaction & logits
        feat_embed = feat_embed.flatten(start_dim=1)
        if hasattr(self, 'dnn'):
            feat_reps = self.dnn(feat_embed)
            
        if self.config.out_layer == 'final':
            return feat_reps
        elif self.config.out_layer == 'hidden':
            # print('hidden111111111')
            return feat_embed


        logits = self.fc_out(feat_reps)
        
        # Loss
        outputs = self.get_outputs(logits, labels)
        
        return outputs
    def get_pretrained_user_reps(self,input_ids,labels,hist_ids,hist_ratings,hist_mask):
        uids = input_ids[:,0]
        uids_embed = self.embed(uids) #bs*dim
        assert self.config.enable_hist_embed
        hist_embed = self.embed(hist_ids)
        if self.config.enable_rating_embed:
            rating_embed = self.embed(hist_ratings)
            hist_embed = torch.cat([hist_embed, rating_embed], dim=-1)
            hist_embed = self.hist_mapping(hist_embed)
            # avg pooling
            masked_hist = hist_embed*hist_mask.unsqueeze(-1)
            avg_hist = masked_hist.sum(dim=1)/hist_mask.sum(dim=1,keepdim=True)
        user_reps = torch.stack([uids_embed,avg_hist],dim=1)
        user_reps = user_reps.flatten(start_dim=1)
        return user_reps

class DIEN(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "output_dim"]

    def __init__(self, config: Config, gru_type="AUGRU"):
        super(DIEN, self).__init__(model_name="DIEN", config=config)
        
        self.gru_type = gru_type
        self.extraction_module = nn.GRU(input_size=config.embed_size,
                                        hidden_size=config.hidden_size,
                                        batch_first=True)
        if gru_type in ["AGRU", "AUGRU"]:
            self.evolving_module
        else:
            self.evolving_module = nn.GRU(input_size=config.embed_size,
                                          hidden_size=config.hidden_size,
                                          batch_first=True)
        # if gru_type in ["AIGRU", "AGRU", "AUGRU"]:
        #     self.attention_module = DynamicGRU()
        

    def interest_extraction(self, hist_emb, hist_mask):
        hist_lens = hist_mask.sum(dim=1).cpu()
        packed_seq = pack_padded_sequence(hist_emb, 
                                          hist_lens, 
                                          batch_first=True, 
                                          enforce_sorted=False)
        packed_interests, _ = self.extraction_module(packed_seq)
        interest_emb, _ = pad_packed_sequence(packed_interests,
                                              batch_first=True,
                                              padding_value=0.0,
                                              total_length=hist_mask.size(1))
        return packed_interests, interest_emb
    
    def interest_evolution(self, packed_interests, interest_emb, target_emb, mask):
        if self.gru_type == "GRU":
            _, hidden = self.evolving_module(packed_interests)
        else:
            attn_scores = self.attention_module(interest_emb, target_emb, mask)
            hist_lens = mask.sum(dim=1).cpu()
            if self.gru_type == "AIGRU":
                packed_inputs = pack_padded_sequence(interest_emb * attn_scores,
                                                     hist_lens,
                                                     batch_first=True,
                                                     enforce_sorted=False)
                _, hidden = self.evolving_module(packed_inputs)
            else:
                packed_scores = pack_padded_sequence(attn_scores,
                                                     hist_lens,
                                                     batch_first=True,
                                                     enforce_sorted=False)
                _, hidden = self.evolving_module(packed_interests, packed_scores)
        return hidden.squeeze()
    


class GRU4Rec(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "hidden_act",
                   "output_dim", "num_gru_layers"]

    def __init__(self, config: Config):
        super(GRU4Rec, self).__init__(model_name="GRU4Rec", config=config)

        self.embed = Embeddings(config)
        self.gru = nn.GRU(input_size=config.embed_size,
                          hidden_size=config.embed_size,
                          num_layers=config.num_gru_layers,
                          batch_first=True)
        self.dnn = MLPBlock(
            input_dim=config.num_fields * config.embed_size, 
            hidden_size=config.hidden_size, 
            num_hidden_layers=config.num_hidden_layers, 
            hidden_dropout_rate=config.hidden_dropout_rate, 
            hidden_act=config.hidden_act
        )
        self.fc_out = nn.Linear(config.hidden_size, config.output_dim)



    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # Embedding
        feat_embed = self.embed(input_ids)
        hist_embed = self.embed(hist_ids)
        assert self.config.enable_hist_embed, "GRU4Rec NEEDS user history sequence."
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)

        # GRU
        hist_lens = hist_mask.sum(dim=1).cpu()
        packed_seq = pack_padded_sequence(hist_embed, hist_lens, batch_first=True, enforce_sorted=False)
        _, final_hidden = self.gru(packed_seq)
        h_n = final_hidden[-1,:,:].unsqueeze(1)

        # Logits
        feat_embed = torch.cat([feat_embed, h_n], dim=1)
        dnn_output = self.dnn(feat_embed.flatten(start_dim=1))
        logits = self.fc_out(dnn_output)

        # Loss
        outputs = self.get_outputs(logits, labels)
        
        return outputs



class SASRec(BaseModel):
    used_params = ["embed_size", "num_attn_layers", "attn_size", "num_attn_heads", 
                "attn_probs_dropout_rate", "use_lr", "res_conn", "attn_scale", "output_dim", 
                "dnn_size", "num_dnn_layers", "dnn_act", "dnn_drop"]
    
    def __init__(self, config: Config):
        super(SASRec, self).__init__(model_name="SASRec", config=config)

        self.embed = Embeddings(config)
        self.self_attention = nn.ModuleList(
            [MultiHeadSelfAttention(config.embed_size if i == 0 else config.num_attn_heads * config.attn_size,
                                     attention_dim=config.attn_size, 
                                     num_heads=config.num_attn_heads, 
                                     dropout_rate=config.attn_probs_dropout_rate, 
                                     use_residual=config.res_conn, 
                                     use_scale=config.attn_scale, 
                                     layer_norm=False,
                                     align_to="output") 
             for i in range(config.num_attn_layers)])
        attn_dim = config.attn_size * config.num_attn_heads

        final_dim = attn_dim + (config.num_fields - 1) * config.embed_size
        if config.num_dnn_layers > 0:
            self.dnn = MLPBlock(input_dim=final_dim,
                            hidden_size=config.dnn_size,
                            num_hidden_layers=config.num_dnn_layers,
                            hidden_dropout_rate=config.dnn_drop,
                            hidden_act=config.dnn_act)
            final_dim = config.dnn_size
        self.fc_out = nn.Linear(final_dim, config.output_dim)

    
    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # Embedding
        feat_embed = self.embed(input_ids)
        assert self.config.enable_hist_embed
        hist_embed = self.embed(hist_ids)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)

        attention_mask = (hist_mask.to(torch.bool))
        attention_input = hist_embed
        for attention_block in self.self_attention:
            attention_out = attention_block(attention_input)
            attention_input = attention_out
        attention_out = attention_out[:, -1, :]

        feat_embed = feat_embed.flatten(start_dim=1)
        feat_embed = torch.cat([feat_embed, attention_out], dim=1)
        if hasattr(self, 'dnn'):
            feat_embed = self.dnn(feat_embed)
        logits = self.fc_out(feat_embed)

        outputs = self.get_outputs(logits, labels)

        return outputs



class Caser(BaseModel):
    used_params = ["embed_size", "hidden_size", "num_hidden_layers", "hidden_dropout_rate", "num_v_kernel", "num_h_kernel"]
    
    def __init__(self, config: Config):
        super(Caser, self).__init__(model_name="Caser", config=config)

        self.L = config.hist_lens
        self.n_v = config.num_v_kernel
        self.n_h = config.num_h_kernel
        self.embed = Embeddings(config)

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1,
                                out_channels=self.n_v,
                                kernel_size=(self.L, 1))

        # horizontal conv layer
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, config.embed_size)) for i in range(1, self.L + 1)])

        final_dim = (config.num_fields - 1 + self.n_v) * config.embed_size + self.n_h * self.L
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=final_dim,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                hidden_dropout_rate=config.hidden_dropout_rate,
                                hidden_act=config.hidden_act)
            final_dim = config.hidden_size
        self.fc_out = nn.Linear(final_dim, config.output_dim)

    
    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # Embedding
        feat_embed = self.embed(input_ids)
        assert self.config.enable_hist_embed, "Caser NEEDS user history sequence."
        hist_embed = self.embed(hist_ids).unsqueeze(1)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)
        
        # Convolution on hist
        v_out = self.conv_v(hist_embed)
        v_out = v_out.view(input_ids.shape[0], -1)

        h_out = []
        for conv in self.conv_h:
            act_out = F.relu(conv(hist_embed)).squeeze(3)
            pooling_out = F.max_pool1d(act_out, act_out.shape[2]).squeeze(2)
            h_out.append(pooling_out)
        h_out = torch.cat(h_out, dim=-1)

        # DNN
        feat_embed = torch.cat([h_out, v_out, feat_embed.flatten(start_dim=1)], dim=-1)
        if hasattr(self, 'dnn'):
            feat_embed = self.dnn(feat_embed)
        logits = self.fc_out(feat_embed)

        # Loss
        outputs = self.get_outputs(logits, labels)
        return outputs
        
            

class ETA(BaseModel):
    used_params = ["hash_bits", "embed_size", "attn_size", "num_attn_heads", 
                   "attn_probs_dropout_rate", "dnn_size", "num_dnn_layers", "dnn_drop"]

    def __init__(self, config: Config):
        super(ETA, self).__init__(model_name="ETA", config=config)

        self.item_field_idx = config.item_field_idx
        self.embed = Embeddings(config)
        self.hash_bits = config.hash_bits
 
        self.short_attention = MultiHeadTargetAttention(
            input_dim=config.embed_size,
            attention_dim=config.attn_size,
            num_heads=config.num_attn_heads,
            dropout_rate=config.attn_probs_dropout_rate,
        )

        self.long_attention = MultiHeadTargetAttention(
            input_dim=config.embed_size,
            attention_dim=config.attn_size,
            num_heads=config.num_attn_heads,
            dropout_rate=config.attn_probs_dropout_rate,
        )

        self.random_rotations = nn.Parameter(torch.randn(config.embed_size, self.hash_bits), requires_grad=False)
        
        final_dim = config.embed_size * (config.num_fields + 1) # hist >> short_interest and long_interest

        if config.num_dnn_layers > 0:
            self.dnn = MLPBlock(input_dim=final_dim,
                            hidden_size=config.dnn_size,
                            num_hidden_layers=config.num_dnn_layers,
                            hidden_dropout_rate=config.dnn_drop)
            final_dim = config.dnn_size
        self.fc_out = nn.Linear(final_dim, config.output_dim)

    
    def forward(self, input_ids, labels=None, hist_ids=None, hist_ratings=None, hist_mask=None):
        # Embeddings
        feat_embed = self.embed(input_ids)
        target_embed = feat_embed[:,self.item_field_idx,:]
        assert self.config.enable_hist_embed
        hist_embed = self.embed(hist_ids)
        if self.config.enable_rating_embed:
            feat_embed = self.get_hist_embed(self.embed, feat_embed, hist_ratings, hist_mask)            

        hist_mask = hist_mask.to(torch.bool)
        
        # Short interest attention
        hist_lens = 60 if self.config.dataset == "BookCrossing" else 30
        short_interest_embed = self.short_attention(target_embed, hist_embed[:,-hist_lens:,:], hist_mask[:,-hist_lens:])

        # long interest attention
        topk_embed, topk_mask = self.topk_retrieval(self.random_rotations, target_embed, hist_embed, hist_mask, hist_lens)
        long_interest_embed = self.long_attention(target_embed, topk_embed, topk_mask)

        feat_embed = torch.cat([feat_embed, short_interest_embed.unsqueeze(1), long_interest_embed.unsqueeze(1)], dim=1)
        feat_embed = feat_embed.flatten(start_dim=1)

        if hasattr(self, 'dnn'):
            feat_embed = self.dnn(feat_embed)
        logits = self.fc_out(feat_embed)
        
        # Loss
        outputs = self.get_outputs(logits, labels)        
        return outputs

        
    def topk_retrieval(self, random_rotations, target_embed, hist_embed, hist_mask, topk=30):
        target_hash = self.lsh_hash(target_embed.unsqueeze(1), random_rotations)
        hist_hash = self.lsh_hash(hist_embed, random_rotations)
        hash_sim = -torch.abs(hist_hash - target_hash).sum(dim=-1)
        hash_sim = hash_sim.masked_fill_(hist_mask.float()==0, -self.hash_bits)
        topk_index = hash_sim.topk(topk, dim=1, largest=True, sorted=True)[1]
        topk_embed = torch.gather(hist_embed, 1, topk_index.unsqueeze(-1).expand(-1, -1, hist_embed.shape[-1]))
        topk_mask = torch.gather(hist_mask, 1, topk_index)
        return topk_embed, topk_mask

    
    def lsh_hash(self, vecs, random_rotations):
        """
        Input: vecs, with shape B x (seq_len) x d
        """
        rotated_vecs = torch.matmul(vecs, random_rotations) # B x seq_len x num_hashes
        hash_code = torch.relu(torch.sign(rotated_vecs))
        return hash_code