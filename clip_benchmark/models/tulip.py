import open_clip
import torch

def load_tulip(model_name: str = "TULIP-so400m-14-384", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model_name,
                                                                pretrained='WebLI',
                                                                cache_dir=cache_dir,)
    state_dict = torch.load(pretrained) # Patch in the delta from the pretrained model
    model.load_state_dict(state_dict, strict=False)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    return model, transform, tokenizer
