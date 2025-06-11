import torch
import torchvision.transforms as T
from transformers import GPT2Tokenizer
from src.model import ImageCaptioningModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {"bos_token": "[start]", "eos_token": "[end]"}
tokenizer.add_special_tokens(special_tokens_dict)


def load_model(model_path):
    model = ImageCaptioningModel(encoder_embed_size=768, prefix_length=20)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Generate captions for the image


def generate_captions(model, image_tensor, beam_width=3, num_captions=3):
    initial_tokens = tokenizer.encode(
        "[start]", return_tensors="pt").to(device)
    beam = [("[start]", 0.0, initial_tokens)]
    completed = []

    for _ in range(20 - 1):
        new_beam = []
        for caption, score, tokens in beam:
            with torch.no_grad():
                img_embed = model.encoder(image_tensor)
                logits = model.decoder(img_embed, tokens)
                logits = model.refinement(logits)
            logits = logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for log_prob, idx in zip(topk_log_probs, topk_indices):
                new_token = idx.unsqueeze(0).unsqueeze(0)
                new_tokens = torch.cat([tokens, new_token], dim=1)
                word = tokenizer.decode([idx.item()]).strip()
                new_caption = caption + " " + word
                new_score = score + log_prob.item()
                if word == "[end]":
                    completed.append((new_caption, new_score))
                else:
                    new_beam.append((new_caption, new_score, new_tokens))
        if not new_beam:
            break
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

    if completed:
        candidate_list = sorted(completed, key=lambda x: x[1], reverse=True)
    else:
        candidate_list = sorted(beam, key=lambda x: x[1], reverse=True)

    captions = [caption.replace("[start]", "").replace("[end]", "").strip()
                for caption, score in candidate_list[:num_captions]]
    return captions
