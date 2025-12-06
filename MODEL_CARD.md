
---

## Modèle de **Model Card** (`MODEL_CARD.md`)

```markdown
# Model Card – Modèle de prédiction du risque d'hospitalisation

## 1. Aperçu du modèle

- **Tâche** : Classification binaire (hospitalisation oui/non)
- **Modèles** : Régression logistique, Random Forest
- **Domaine** : Santé, surveillance hospitalière / triage

## 2. Données

- Population simulée : adultes 18–90 ans
- Variables utilisées :
  - Démographie : âge, sexe
  - Comorbidités : HTA, diabète, cardiopathie
  - Signes vitaux : température, SpO2, fréquence respiratoire
  - Infection respiratoire, statut vaccinal
  - Score de sévérité clinique (severity_score)

## 3. Objectif clinique

Aider à estimer la probabilité d'hospitalisation pour des patients présentant
un syndrome respiratoire, afin de soutenir le triage et prioriser
les cas à surveiller.

## 4. Métriques de performance

- Train/Test split : 80/20, stratifié
- Métriques rapportées :
  - AUC-ROC
  - Accuracy
  - Recall (sensibilité)
  - Précision
  - F1-score
  - Matrice de confusion

## 5. Biais et fairness

- Analyse par sexe (M/F)
- Analyse par tranches d'âge (18–40, 40–60, 60+)
- Risques :
  - Sur-représentation de certaines catégories
  - Décisions cliniques non modélisées (ex: contraintes de lits)

## 6. Limitations

- Données synthétiques → pas de calibration sur données réelles
- Pas d'external validation
- Variables cliniques simplifiées

## 7. Utilisation recommandée

- Usage pédagogique / prototype
- Non adapté pour une décision clinique réelle sans étude approfondie

## 8. Reproductibilité

- Seed fixé à 42
- Code versionné (Git)
- Dataset généré via `src/generate_data.py`
