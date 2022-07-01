def get_score(scores: list[float]) -> float:
    m1 = max(scores)
    scores = scores.copy()
    scores.remove(m1)
    return 0 if m1 == 0 else 1 if len(scores) == 0 else (m1 - max(scores)) / m1
