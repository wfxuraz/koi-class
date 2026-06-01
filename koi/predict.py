def rank_predictions(probs, classes, top_k: int, min_prob: float):
    pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    pairs = [(name, p) for name, p in pairs if p >= min_prob]
    if top_k and top_k > 0:
        pairs = pairs[:top_k]
    return pairs


def format_predictions(ranked) -> str:
    width = max((len(name) for name, _ in ranked), default=0)
    lines = [f"{name:<{width}}  {prob * 100:5.1f}%" for name, prob in ranked]
    return "\n".join(lines)
