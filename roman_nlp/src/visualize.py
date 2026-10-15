import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
PREDICTIONS_PATH = DATA_DIR / "predictions.json"
EMOTIONS_PATH = DATA_DIR / "emotion_profiles.json"

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_LABELS_TR = [
    "Açıklık\n(Openness)",
    "Sorumluluk\n(Conscientiousness)",
    "Dışadönüklük\n(Extraversion)",
    "Uyumluluk\n(Agreeableness)",
    "Nevrotiklik\n(Neuroticism)"
]

EMOTION_LABELS_TR = {
    "joy": "Neşe",
    "sadness": "Hüzün",
    "anger": "Öfke / Çatışma",
    "fear": "Korku / Kaygı",
    "trust_affection": "Şefkat / Güven",
    "curiosity": "Merak / Gözlem"
}

CHAR_NAMES_TR = {
    "Feride": "Feride (Çalıkuşu)",
    "Kamran": "Kâmran",
    "Hayrullah_Bey": "Hayrullah Bey",
    "Mujgan": "Müjgân",
    "Haci_Kalfa": "Hacı Kalfa",
    "Munise": "Munise"
}


def plot_ocean_radar_grid(gt, preds, out_path):
    """6 karakter için Ground Truth vs Model Tahmini Radar Grafiği üretir."""
    characters = [c for c in gt.keys() if c in preds]
    n_chars = len(characters)
    angles = np.linspace(0, 2 * np.pi, len(TRAITS), endpoint=False).tolist()
    angles += angles[:1]  # Beşgeni kapatmak için

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, char_name in enumerate(characters):
        ax = axes[idx]
        y_true = [gt[char_name][t] for t in TRAITS]
        y_pred = [preds[char_name][t] for t in TRAITS]
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

        y_true += y_true[:1]
        y_pred += y_pred[:1]

        ax.plot(angles, y_true, "o-", linewidth=2.2, label="Referans (Ground Truth)", color="#2ca02c")
        ax.fill(angles, y_true, alpha=0.18, color="#2ca02c")

        ax.plot(angles, y_pred, "s--", linewidth=2.2, label="Model Tahmini (BERTurk+LIWC)", color="#1f77b4")
        ax.fill(angles, y_pred, alpha=0.22, color="#1f77b4")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(TRAIT_LABELS_TR, size=8.5, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
        ax.set_title(f"{CHAR_NAMES_TR.get(char_name, char_name)}\n(MAE: {mae:.4f})", size=11, fontweight="bold", pad=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=11)
    fig.suptitle("ÇALIKUŞU-NLP: Karakter Bazlı Big Five (OCEAN) Kişilik Profili Doğrulaması", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_emotion_bars(emotions, out_path):
    """Karakter bazlı Ekman 6 sınıf duygu dağılım grafiği üretir."""
    characters = list(emotions.keys())
    emo_keys = list(EMOTION_LABELS_TR.keys())
    x = np.arange(len(characters))
    width = 0.13

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#f1c40f", "#3498db", "#e74c3c", "#9b59b6", "#2ecc71", "#e67e22"]

    for i, emo_key in enumerate(emo_keys):
        vals = [emotions[c].get(emo_key, 0.0) * 100 for c in characters]
        ax.bar(x + (i - 2.5) * width, vals, width, label=EMOTION_LABELS_TR[emo_key], color=colors[i])

    ax.set_ylabel("Duygu Yoğunluğu (%)", fontsize=11, fontweight="bold")
    ax.set_title("Aşama 5.0: Karakter Bazlı Replik Duygu Dağılım Histogramı", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CHAR_NAMES_TR.get(c, c) for c in characters], fontsize=10, fontweight="bold")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(ncol=3, loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    if not PREDICTIONS_PATH.exists() or not GROUND_TRUTH_PATH.exists():
        print("[!] Önce predict_personality.py çalıştırılmalıdır!")
        return

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radar_path = OUTPUT_DIR / "ocean_radar_comparison.png"
    plot_ocean_radar_grid(gt, preds, radar_path)
    print(f"[*] OCEAN Radar Karşılaştırma Grafiği oluşturuldu : {radar_path}")

    if EMOTIONS_PATH.exists():
        with open(EMOTIONS_PATH, "r", encoding="utf-8") as f:
            emotions = json.load(f)
        emo_path = OUTPUT_DIR / "emotion_distributions.png"
        plot_emotion_bars(emotions, emo_path)
        print(f"[*] Duygu Dağılım Grafiği oluşturuldu             : {emo_path}")


if __name__ == "__main__":
    main()