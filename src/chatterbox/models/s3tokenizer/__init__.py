from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SOS, EOS, and any tokens outside the valid vocab range [0, SPEECH_VOCAB_SIZE-1]"""
    # 1. Flatten to 1D if needed
    if x.ndim > 1:
        x = x.view(-1)
    
    # 2. Find range between first SOS and first EOS if they exist
    s, e = 0, len(x)
    sos_indices = (x == SOS).nonzero(as_tuple=True)[0]
    if len(sos_indices) > 0:
        s = sos_indices[0].item() + 1
    
    eos_indices = (x == EOS).nonzero(as_tuple=True)[0]
    if len(eos_indices) > 0:
        # Search for EOS after the SOS
        valid_eos = eos_indices[eos_indices >= s]
        if len(valid_eos) > 0:
            e = valid_eos[0].item()

    x = x[s:e]
    
    # 3. Aggressively filter out ANY token >= SPEECH_VOCAB_SIZE
    # (T3 can sometimes generate garbage tokens in the 6563-8192 range)
    x = x[x < SPEECH_VOCAB_SIZE]
    
    return x
