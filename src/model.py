import torch
import torch.nn as nn
import timm
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {"bos_token": "[start]", "eos_token": "[end]"}
tokenizer.add_special_tokens(special_tokens_dict)


class ViTEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ViTEncoder, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.reset_classifier(0)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.vit.num_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.vit(images)
        proj = self.fc(features)
        bn_out = self.bn(proj)
        return bn_out


class GPT2Decoder(nn.Module):
    def __init__(self, encoder_embed_size, prefix_length=10, gpt2_model_dir="gpt2", unfreeze_last=4):
        super().__init__()
        self.config = GPT2Config.from_pretrained(gpt2_model_dir)
        self.config.vocab_size = len(tokenizer)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            gpt2_model_dir,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        self.gpt2.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        for param in self.gpt2.parameters():
            param.requires_grad = False
        total_layers = len(self.gpt2.transformer.h)
        for i in range(total_layers - unfreeze_last, total_layers):
            for param in self.gpt2.transformer.h[i].parameters():
                param.requires_grad = True
        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad = True

        self.prefix_length = prefix_length
        self.embedding_proj = nn.Linear(
            encoder_embed_size, self.config.n_embd * prefix_length)

    def forward(self, image_features, input_ids, attention_mask=None):
        batch_size = image_features.size(0)
        prefix = self.embedding_proj(image_features)
        prefix = prefix.view(
            batch_size, self.prefix_length, self.config.n_embd)
        gpt2_inputs = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix, gpt2_inputs], dim=1)
        if attention_mask is None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length, dtype=torch.long, device=inputs_embeds.device)
            attention_mask = torch.cat(
                [prefix_mask, torch.ones_like(input_ids)], dim=1)
        outputs = self.gpt2(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask)
        return outputs.logits


class Refinement(nn.Module):
    def __init__(self, vocab_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vocab_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_dim)
        )

    def forward(self, logits):
        refined = self.mlp(logits)
        return logits + refined


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder_embed_size, prefix_length, gpt2_model_dir="gpt2", augment=None):
        super().__init__()
        self.encoder = ViTEncoder(encoder_embed_size)
        self.decoder = GPT2Decoder(
            encoder_embed_size, prefix_length, gpt2_model_dir)
        self.refinement = Refinement(self.decoder.config.vocab_size)
        self.augment = augment

    def forward(self, images, input_ids, attention_mask=None, training=False):
        if training and self.augment is not None:
            images = self.augment(images)
        image_features = self.encoder(images)
        logits = self.decoder(image_features, input_ids, attention_mask)
        refined_logits = self.refinement(logits)
        return refined_logits
