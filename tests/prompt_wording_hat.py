"""The headphones left the classroom scene: the voice prompts name the hat as the example
wearable now. The tests that compare a session or tool list byte for byte with an older base commit apply this one
intended wording change to the base text (every other byte must still match)."""
HAT_WORDING = (("(headphones, chair", "(hat, chair"), ("(headphones)", "(hat, sunglasses)"),
                 ("ball, headphones, chair", "ball, hat, chair"), ("put on headphones,", "put on the hat,"))


def hat_wording(s: str) -> str:
    for a, b in HAT_WORDING:
        s = s.replace(a, b)
    return s
