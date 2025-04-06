import numpy as np

def low_risk(prob):
    if prob <= 0.3:
        return 1
    elif 0.3 < prob <= 0.5:
        return (0.5 - prob) / 0.2
    else:
        return 0

def medium_risk(prob):
    if 0.3 < prob < 0.5:
        return (prob - 0.3) / 0.2
    elif 0.5 <= prob <= 0.7:
        return (0.7 - prob) / 0.2
    else:
        return 0

def high_risk(prob):
    if prob <= 0.5:
        return 0
    elif 0.5 < prob < 0.7:
        return (prob - 0.5) / 0.2
    else:
        return 1

def fuzzy_infer(prob):
    if isinstance(prob, np.ndarray):
        return [fuzzy_infer_single(p) for p in prob]
    else:
        return fuzzy_infer_single(prob)

def fuzzy_infer_single(prob):
    low = low_risk(prob)
    med = medium_risk(prob)
    high = high_risk(prob)

    fuzzy_result = (low * 0.2 + med * 0.5 + high * 0.8) / (low + med + high + 1e-6)
    return round(fuzzy_result, 3)

def fuzzy_predict(prob_list):
    return [fuzzy_infer(p) for p in prob_list]
