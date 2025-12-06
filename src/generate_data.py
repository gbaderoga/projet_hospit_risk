import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N = 2000  # nombre de lignes

def generate_synthetic_hospit_data(n_samples: int = N) -> pd.DataFrame:
    ages = np.random.randint(18, 90, size=n_samples)
    sexes = np.random.choice(['F', 'M'], size=n_samples, p=[0.55, 0.45])

    # comorbidités (0/1)
    hta = np.random.binomial(1, p=0.35, size=n_samples)
    diabete = np.random.binomial(1, p=0.20, size=n_samples)
    cardiopathie = np.random.binomial(1, p=0.10, size=n_samples)

    # variables cliniques
    temp = np.round(np.random.normal(loc=37.3, scale=0.7, size=n_samples), 1)
    spo2 = np.clip(np.round(np.random.normal(loc=95, scale=3, size=n_samples)), 80, 100)
    freq_resp = np.clip(np.round(np.random.normal(loc=20, scale=4, size=n_samples)), 10, 40)

    # statut infection respiratoire (ex: grippe / covid-like)
    infection = np.random.binomial(1, p=0.4, size=n_samples)
    vacc = np.random.binomial(1, p=0.5, size=n_samples)  # vacciné oui/non

    # score de sévérité (NEWS2-like simplifié)
    severity_score = (
        (temp > 38.5).astype(int)
        + (spo2 < 93).astype(int)
        + (freq_resp > 24).astype(int)
        + hta
        + diabete
        + cardiopathie
    )

    # probabilité d’hospitalisation en fonction des facteurs
    logit = (
        -4
        + 0.03 * (ages - 50)
        + 0.8 * hta
        + 1.0 * diabete
        + 1.3 * cardiopathie
        + 0.9 * infection
        + 0.15 * (severity_score)
        - 0.5 * vacc
    )
    prob_hospit = 1 / (1 + np.exp(-logit))
    hospit = np.random.binomial(1, p=prob_hospit)

    df = pd.DataFrame({
        "age": ages,
        "sexe": sexes,
        "hta": hta,
        "diabete": diabete,
        "cardiopathie": cardiopathie,
        "temperature": temp,
        "spo2": spo2,
        "freq_resp": freq_resp,
        "infection_respi": infection,
        "vaccinated": vacc,
        "severity_score": severity_score,
        "hospitalisation": hospit,
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_hospit_data()
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hospit_risk_dataset.csv", index=False)
    print(f"Dataset sauvegardé dans {out_dir / 'hospit_risk_dataset.csv'}")
