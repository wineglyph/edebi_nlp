import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Script'in bulunduğu konuma göre proje ana dizinini (roman_nlp) bulur:
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GT_PATH = BASE_DIR / "data" / "ground_truth.json"

def load_ground_truth(filepath=None):
    if filepath is None:
        filepath = DEFAULT_GT_PATH
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dosya bulunamadı! Aranan tam yol: {filepath.resolve()}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_predictions(predictions, ground_truth):
    results = {}
    all_y_true = []
    all_y_pred = []

    print("\n" + "="*50)
    print("KARAKTER BAZLI KİŞİLİK DEĞERLENDİRME RAPORU")
    print("="*50)

    for char_name, true_traits in ground_truth.items():
        if char_name not in predictions:
            continue

        pred_traits = predictions[char_name]
        y_true = np.array([true_traits[t] for t in TRAITS])
        y_pred = np.array([pred_traits[t] for t in TRAITS])

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        cos_sim = 1.0 - cosine(y_true, y_pred) if np.linalg.norm(y_pred) > 0 else 0.0

        results[char_name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "Cosine_Similarity": round(float(cos_sim), 4)
        }

        print(f"\nKarakter: {char_name}")
        print(f"  - MAE (Ortalama Mutlak Hata) : {results[char_name]['MAE']}")
        print(f"  - RMSE (Hata Karekökü)       : {results[char_name]['RMSE']}")
        print(f"  - Profil Benzerliği (Cosine) : {results[char_name]['Cosine_Similarity']}")

    overall_mae = mean_absolute_error(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_cos_sim = 1.0 - cosine(all_y_true, all_y_pred)
    spearman_corr, _ = spearmanr(all_y_true, all_y_pred)

    print("\n" + "-"*50)
    print("GENEL MODEL BAŞARISI (TÜM KARAKTERLER)")
    print(f"  - Toplam MAE                 : {round(float(overall_mae), 4)} (Hedef: < 0.15)")
    print(f"  - Toplam RMSE                : {round(float(overall_rmse), 4)}")
    print(f"  - Toplam Kosinüs Benzerliği  : {round(float(overall_cos_sim), 4)} (Hedef: >= 0.85)")
    print(f"  - Spearman Sıralama Korel.   : {round(float(spearman_corr), 4)}")
    print("="*50 + "\n")

    return results

if __name__ == "__main__":
    gt = load_ground_truth()
    
    dummy_predictions = {
        "Feride": {
            "openness": 0.80, "conscientiousness": 0.60,
            "extraversion": 0.85, "agreeableness": 0.45, "neuroticism": 0.70
        },
        "Kamran": {
            "openness": 0.50, "conscientiousness": 0.40,
            "extraversion": 0.55, "agreeableness": 0.70, "neuroticism": 0.60
        },
        "Hayrullah_Bey": {
            "openness": 0.75, "conscientiousness": 0.85,
            "extraversion": 0.70, "agreeableness": 0.65, "neuroticism": 0.25
        }
    }

    evaluate_predictions(dummy_predictions, gt)