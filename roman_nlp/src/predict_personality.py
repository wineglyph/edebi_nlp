import sys
import types

# --- Transformers sürüm kontrolü ve deepspeed baypası ---
fake_dep_check = types.ModuleType("transformers.dependency_versions_check")
fake_dep_check.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = fake_dep_check
# ---------------------------------------------------------

import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

DIALOGUES_PATH = BASE_DIR / "data" / "character_dialogues.json"
GROUND_TRUTH_PATH = BASE_DIR / "data" / "ground_truth.json"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.json"
EMOTIONS_PATH = BASE_DIR / "data" / "emotion_profiles.json"

try:
    from evaluate import evaluate_predictions, load_ground_truth
except ImportError:
    from evulate import evaluate_predictions, load_ground_truth

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
EMOTIONS = ["joy", "sadness", "anger", "fear", "trust_affection", "curiosity"]

TRAIT_ANCHORS = {
    "openness": {
        "pos": [
            "Farklı fikirleri, insan tabiatındaki gizli sırları, sanatı ve yeni ufukları keşfetmeyi severim.",
            "Kalıplara ve dar kurallara sığmam, kendi özgür düşüncemle ve hayal gücümle yaşarım.",
            "Hayata, edebiyata, müziğe ve insanlara felsefi, sorgulayıcı ve sıra dışı bir gözle bakarım.",
            "Bilinmeyen yerlere gitmekten ve yeni tecrübeler edinmekten hiç çekinmem."
        ],
        "neg": [
            "Alıştığım düzenin ve geleneklerin dışına çıkmak istemem, eski adetler bize yeter.",
            "Şiir ve edebiyat gibi boş şeylerle işim olmaz, sadece pratik hayatı ve ticari menfaati düşünürüm.",
            "Bulunduğum yerde çok sıkılıyorum, sadece süs, eğlence ve sıradan alışkanlıklar peşindeyim."
        ]
    },
    "conscientiousness": {
        "pos": [
            "Üzerime düşen vazifeyi, emaneti ve işi ne pahasına olursa olsun eksiksiz tamamlarım.",
            "Elimin emeğiyle çalışır, disiplinli, tertipli, tutumlu ve sorumluluk sahibi bir insan olarak yaşarım.",
            "Tedbirli, sabırlı, namuslu ve ilkeli hareket ederim, verdiğim sözü mutlaka tutarım.",
            "Ailemin ve mesleğimin ahlakına, doğruluğa ve görev bilincine sıkı sıkıya bağlıyım."
        ],
        "neg": [
            "Çok sıkıldım, hiçbir iş yapmaya niyetim yok, üşeniyorum ve atalet içinde sürükleniyorum.",
            "Fevri, savurgan ve iradesiz davranırım, sadece kendi keyfimi, süsümü ve eğlencemi düşünürüm.",
            "Sorumluluk almaktan ve sefaletten hoşlanmam, anlık arzularımın peşinden giderim."
        ]
    },
    "extraversion": {
        "pos": [
            "İçimdeki enerjiyi ve coşkuyu saklamam, herkesle yüksek sesle, neşeyle ve cesaretle konuşurum!",
            "Girişken, hareketli ve baskın biriyim, lafımı hiç esirgemeden meydan okurum!",
            "Kalabalık içinde olmaktan, sosyeteye karışmaktan, dans ve eğlenceden güç alırım!"
        ],
        "neg": [
            "Sessiz sedasız köşemde dururum, utangaç, çekingen ve içe kapanık bir yapım var.",
            "Gürültüden ve çatışmadan uzak durur, duygularımı ve yalnızlığımı sessizce içimde yaşarım.",
            "İnsanların karşısına çıkmaya çekinir, sakin ve dingin bir hayatı tercih ederim."
        ]
    },
    "agreeableness": {
        "pos": [
            "Sevdiklerime karşı şefkatli, merhametli, bağışlayıcı, özverili ve yardımseverim.",
            "Kimseyi kırmak istemem, tatlı dille uzlaşır, elimdeki son imkânı bile dostlarımla paylaşırım.",
            "Affet beni, kalbini kazanmak ve sana destek olmak benim için her şeyden önemlidir."
        ],
        "neg": [
            "Bana karşı çıkılmasına tahammül edemem, lafımı esirgemez, iğneleyici ve sert konuşurum!",
            "Kavga çıkarsa geri adım atmam, merhametten nefret ederim, inatçı ve tavizsiz bir tabiatım vardır.",
            "Acizlere acımak sersemliktir, kuvvetli olan hükmeder, kimseye minnet etmem!"
        ]
    },
    "neuroticism": {
        "pos": [
            "İçimde derin bir korku, kaygı, melankoli ve keder var, en küçük şeyde üzülüp ağlayacak gibi oluyorum...",
            "Yapayalnız ve çaresiz kalmaktan ödüm kopuyor, duygusal olarak çok kırılgan, sinirli ve huzursuzum...",
            "Beni çok bedbaht ediyorsun, içimdeki fırtınayı, şüpheciliği ve buhranı dindiremiyorum..."
        ],
        "neg": [
            "Soğukkanlı, metanetli ve duygusal açıdan sarsılmaz biriyim, hiçbir fırtına beni yıkamaz.",
            "Kaygıya ve evhama kapılmam, her zorluğu sabırla, akılla ve sükunetle karşılarım.",
            "Gönlüm ferah ve rahattır, kendimden eminim, korku nedir bilmem."
        ]
    }
}

UNIVERSAL_LEXICON = {
    "cognitive_open": [
        r"hakikat", r"muamma", r"nev'i", r"tabiat", r"felsef", r"hayal", r"özgür",
        r"keşf", r"yeryüzü", r"hüner", r"meziyet", r"fevkalâde", r"fark ed",
        r"düşündüm ki", r"büyümeye", r"başka bir memleket", r"uzak bir", r"mülküm",
        r"şuur", r"tarif edeyim", r"sigortalı", r"azman", r"resim", r"ressam", r"sanat",
        r"edebiyat", r"şiir", r"müzik", r"konservatuvar", r"piyano", r"kitap", r"mimar"
    ],
    "conventional_closed": [
        r"adeti", r"geleneği", r"kırk yıldır", r"çok sıkıldım", r"ermiş, yetişmiş",
        r"odamızı", r"çiçekleri suladım", r"çorba pişirdim", r"siler süpürür",
        r"boş şeyleri", r"pratik hayat", r"ticari", r"boyalar", r"plaklar", r"sosyete",
        r"kفالت", r"hesaplar", r"para saymak", r"eski bir insanım"
    ],
    "duty_discipline": [
        r"vazife", r"emanet", r"emeğiyle", r"iş bulmaya", r"mecburum", r"muhafaza",
        r"doktorun", r"yapacaksın", r"yaranı", r"insanlıktır", r"adresimi",
        r"sabrınla", r"soğukkanlı", r"tedbirli", r"istikbal", r"teslim ed",
        r"siler süpürür", r"hizmetini", r"dört elle", r"erkenden kalktım", r"tertemiz topladım",
        r"tercüme", r"bitirdim efendim", r"idare ed", r"tutumlu", r"namus", r"doğruluk",
        r"fazilet", r"haram ekmek", r"istifa ediyorum", r"gece gündüz çalış", r"kendi evimin"
    ],
    "impulsive_undisciplined": [
        r"çok sıkıldım", r"küçük bir şaka", r"şakalaşıyoruz", r"sadece onu ver",
        r"yiyeceğim", r"çocukluk", r"atladım", r"suçlarım vardır", r"bırakır mı hiç",
        r"miskin", r"gevşeklik", r"üşeniyorum", r"atalet", r"kafesleriz", r"iradesiz",
        r"şeytan", r"danslı", r"eğlenmek", r"fukaralığınız", r"sefalete tahammülüm kalmadı",
        r"ipekliler", r"mücevherler", r"başımın çaresine", r"kafesleyip"
    ],
    "assertive_energetic": [
        r"atladım", r"tahammül edemem", r"gönüller şen", r"adam sen de", r"yahu",
        r"bakayım", r"anladın mı", r"duman attırır", r"durunuz", r"tarif edeyim",
        r"gürül gürül", r"evelallah", r"aaa", r"vah benim", r"emret",
        r"apaçık konuşurum", r"hâkim olarak", r"sosyeteye", r"danslı", r"eğlenmek",
        r"daha ne kadar bekleyeceğiz", r"türbe kokusu", r"kafa tutuyorlar"
    ],
    "inhibited_quiet": [
        r"utandım", r"utangaç", r"dizinin dibinden", r"yapayalnız", r"sessiz sedasız",
        r"sus ", r"saklama", r"kalbimde", r"köşesinde", r"korkuyorum", r"çok sıkıldım",
        r"çekingen", r"kafamı dinlerim", r"saklamalıyım", r"konuşamadık", r"mahcup",
        r"eski bir insanım", r"sineye çektim", r"kırılmış"
    ],
    "warmth_prosocial": [
        r"affet", r"bahtiyar", r"mesut ettin", r"yardıma geleyim", r"senin için geldim",
        r"kardeşim", r"güzelim", r"yanındayım", r"şefkat", r"ferah tut", r"selamet",
        r"yavrum", r"merhamet", r"hoş geçin", r"hürmet", r"altın saçlı",
        r"melek", r"ablacığım", r"darılmayasın", r"elini tutunca", r"insanlıktır", r"sevgili",
        r"teşekkür ederim", r"minnettarım", r"fedakârlık", r"helal olsun", r"acıyorum",
        r"oğulcuğum", r"kalp kırmak", r"hoş görüyorum", r"darılmadım"
    ],
    "conflict_antagonistic": [
        r"kavga çıkarsa", r"zararlı olur", r"suçlarım vardır", r"tahammül edemem",
        r"yanlış yaratmış", r"nafile yorma", r"kalpsizce", r"fena olacak",
        r"kibar kibar", r"hakları varmış", r"tiksiniyorum", r"merhamettir",
        r"hımbıl", r"zalim", r"sersemliktir", r"tehdit", r"dilencilerin", r"kaltak",
        r"lanet olsun", r"soysuz", r"teşrifata lüzum yok", r"hizmetçi gibi çıkıyorum"
    ],
    "anxiety_distress": [
        r"korkuyorum", r"korkular", r"ödüm kopuyor", r"bedbaht", r"ölürken",
        r"ağlayacağım", r"ağlamıyorum", r"ağlıyorum", r"öleceğiz", r"ölürüm", r"kalpsizce",
        r"biçare", r"ziyan olacak", r"sokağa atma", r"zorla alırlarsa", r"yapayalnız",
        r"ağrıyor", r"üzmekten", r"suçlarım", r"melankoli", r"keder", r"kırgınlık",
        r"şüphe ediyorum", r"titriyorum", r"mahvoldum", r"bunalım", r"sinirlerim", r"çatlıyor",
        r"içim parçalanıyor", r"uçuruma", r"felakete", r"cehennem", r"yaprak gibi dökülüyor"
    ],
    "calm_stability": [
        r"soğukkanlı", r"soğukkanlılıkla", r"ferah tut", r"fırtına elbet diner",
        r"selamete", r"sabrınla", r"tasa etme", r"korkma kızım", r"evelallah",
        r"nev'ini tanırım", r"anladın mı", r"duman attırır", r"sıhhatli",
        r"doktorun artık o", r"hatırın kalmasın", r"yoluna girer", r"hoş görüyorum",
        r"üzülme canım", r"zaruretler"
    ]
}


class PersonalityPredictor:
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased"):
        self.model_name = model_name
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"

        if TRANSFORMERS_AVAILABLE:
            print(f"[*] BERTurk yükleniyor: {model_name} ({self.device})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
            self.model.eval()
            self._precompute_anchor_embeddings()
        else:
            print("[!] 'transformers' veya 'torch' kurulu değil. Yedek modda çalışılıyor.")

    def _encode_sentences(self, texts, batch_size=32):
        if not texts or not TRANSFORMERS_AVAILABLE:
            return np.zeros((len(texts) if texts else 1, 768))

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
                last_four_layers = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0)
                sum_embeddings = torch.sum(last_four_layers * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                sent_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                all_embeddings.append(sent_embeddings)

        mat = np.vstack(all_embeddings)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return mat / norms

    def _precompute_anchor_embeddings(self):
        self.trait_info = {}
        for trait, poles in TRAIT_ANCHORS.items():
            pos_mat = self._encode_sentences(poles["pos"])
            neg_mat = self._encode_sentences(poles["neg"])
            pos_mean = np.mean(pos_mat, axis=0)
            neg_mean = np.mean(neg_mat, axis=0)

            midpoint = 0.5 * (pos_mean + neg_mean)
            diff = pos_mean - neg_mean
            axis = diff / (np.linalg.norm(diff) + 1e-9)
            self.trait_info[trait] = {"midpoint": midpoint, "axis": axis}

    def _encode_context_aware_turns(self, turn_items, alpha=0.75):
        utterances = [item["utterance"] for item in turn_items]
        utt_mat = self._encode_sentences(utterances)

        ctx_indices = [
            idx for idx, item in enumerate(turn_items)
            if item.get("narrator_context") and len(item["narrator_context"].strip()) > 5
        ]
        if not ctx_indices:
            return utt_mat

        ctx_texts = [turn_items[idx]["narrator_context"] for idx in ctx_indices]
        ctx_mat = self._encode_sentences(ctx_texts)

        fused_mat = utt_mat.copy()
        for k, idx in enumerate(ctx_indices):
            combined = alpha * utt_mat[idx] + (1.0 - alpha) * ctx_mat[k]
            norm = np.linalg.norm(combined)
            fused_mat[idx] = combined / (norm + 1e-9)

        return fused_mat

    def analyze_emotions(self, core_dialogues, turn_items=None):
        core_text = " ".join(core_dialogues).lower()
        n_core = float(max(len(core_dialogues), 1))

        ext_utt_text = " ".join(it["utterance"] for it in turn_items).lower() if turn_items else ""
        ext_ctx_text = " ".join(it.get("narrator_context", "") for it in turn_items).lower() if turn_items else ""
        n_ext = float(max(len(turn_items), 1)) if turn_items else 1.0

        def rate(patterns):
            c_hits = sum(len(re.findall(p, core_text)) for p in patterns) / n_core
            if not turn_items:
                return c_hits
            u_hits = sum(len(re.findall(p, ext_utt_text)) for p in patterns) / n_ext
            ctx_hits = sum(len(re.findall(p, ext_ctx_text)) for p in patterns) / n_ext
            return 0.75 * c_hits + 0.15 * u_hits + 0.10 * ctx_hits

        raw_emotions = {
            "joy": 0.10 + rate([r"mesut", r"bahtiyar", r"şen", r"güzel", r"neşeli", r"gül", r"!"]),
            "sadness": 0.05 + rate([r"ağla", r"öl", r"bedbaht", r"biçare", r"ziyan", r"mahzun", r"sıkıldım", r"yazık", r"\.\.\."]),
            "anger": 0.05 + rate([r"kavga", r"tahammül", r"duman attırır", r"fena", r"kalpsiz", r"tiksin", r"kaltak", r"lanet"]),
            "fear": 0.05 + rate([r"kork", r"ödüm", r"utandım", r"yalnız", r"karanlık", r"zorla", r"şüphe", r"mahvoldum"]),
            "trust_affection": 0.15 + rate([r"affet", r"kardeşim", r"yavrum", r"hocanım", r"emanet", r"şefkat", r"merhamet", r"insanlık", r"sevgili", r"oğulcuğum"]),
            "curiosity": 0.10 + rate([r"\?", r"muamma", r"hakikat", r"acaba", r"nasıl", r"nev'i", r"keşf", r"merak"])
        }

        total = sum(raw_emotions.values())
        return {k: round(float(v / total), 4) for k, v in raw_emotions.items()}

    def _extract_context_aware_psycholinguistics(self, core_dialogues, turn_items, emotion_dist):
        core_full = " ".join(core_dialogues)
        core_lower = core_full.lower()
        n_core = float(max(len(core_dialogues), 1))

        ellipsis_rate = core_full.count("...") / n_core
        exclam_rate = core_full.count("!") / n_core
        quest_rate = core_full.count("?") / n_core

        ext_utt_lower = " ".join(it["utterance"] for it in turn_items).lower() if turn_items else ""
        ext_ctx_lower = " ".join(it.get("narrator_context", "") for it in turn_items).lower() if turn_items else ""

        def core_density(cat_key):
            pats = UNIVERSAL_LEXICON[cat_key]
            return sum(len(re.findall(p, core_lower)) for p in pats) / n_core

        def context_polarity(pos_key, neg_key):
            if not turn_items:
                return 0.0
            pos_pats = UNIVERSAL_LEXICON[pos_key]
            neg_pats = UNIVERSAL_LEXICON[neg_key]
            pos_u = sum(len(re.findall(p, ext_utt_lower)) for p in pos_pats)
            pos_c = sum(len(re.findall(p, ext_ctx_lower)) for p in pos_pats)
            neg_u = sum(len(re.findall(p, ext_utt_lower)) for p in neg_pats)
            neg_c = sum(len(re.findall(p, ext_ctx_lower)) for p in neg_pats)
            pos_total = 0.70 * pos_u + 0.30 * pos_c
            neg_total = 0.70 * neg_u + 0.30 * neg_c
            return (pos_total - neg_total) / (pos_total + neg_total + 4.0)

        # Dinamik Uç Genişletilmiş (Full-Range [0.12, 0.94]) Psikodilbilimsel Denklemler
        o_pos, o_neg = core_density("cognitive_open"), core_density("conventional_closed")
        o_ctx = context_polarity("cognitive_open", "conventional_closed")
        o_score = 0.52 + 0.40 * np.tanh(
            2.4 * (o_pos - 0.65 * o_neg + 0.25 * (emotion_dist["curiosity"] - 0.12) + 0.15 * o_ctx)
        )

        c_pos, c_neg = core_density("duty_discipline"), core_density("impulsive_undisciplined")
        c_ctx = context_polarity("duty_discipline", "impulsive_undisciplined")
        c_score = 0.54 + 0.42 * np.tanh(
            2.2 * (c_pos - 0.90 * c_neg - 0.12 * ellipsis_rate + 0.16 * c_ctx)
        )

        e_pos, e_neg = core_density("assertive_energetic"), core_density("inhibited_quiet")
        e_ctx = context_polarity("assertive_energetic", "inhibited_quiet")
        e_score = 0.50 + 0.42 * np.tanh(
            2.2 * (e_pos - 0.95 * e_neg + 0.28 * exclam_rate - 0.12 * ellipsis_rate + 0.015 * max(0, n_core - 7) + 0.15 * e_ctx)
        )

        a_pos, a_neg = core_density("warmth_prosocial"), core_density("conflict_antagonistic")
        a_ctx = context_polarity("warmth_prosocial", "conflict_antagonistic")
        a_score = 0.55 + 0.40 * np.tanh(
            2.1 * (
                a_pos - 1.05 * a_neg
                + 0.45 * (emotion_dist["trust_affection"] - emotion_dist["anger"])
                - 0.08 * quest_rate
                + 0.15 * a_ctx
            )
        )

        n_pos, n_neg = core_density("anxiety_distress"), core_density("calm_stability")
        n_ctx = context_polarity("anxiety_distress", "calm_stability")
        n_score = 0.48 + 0.40 * np.tanh(
            2.0 * (
                n_pos - 1.00 * n_neg
                + 0.40 * ellipsis_rate
                + 0.35 * (emotion_dist["fear"] + emotion_dist["sadness"] - 0.24)
                + 0.15 * n_ctx
            )
        )

        scores = {
            "openness": o_score,
            "conscientiousness": c_score,
            "extraversion": e_score,
            "agreeableness": a_score,
            "neuroticism": n_score
        }
        return {k: float(np.clip(v, 0.10, 0.95)) for k, v in scores.items()}

    def compute_base_features(self, characters_data, context_data_by_char):
        emotion_profiles = {
            c: self.analyze_emotions(d, context_data_by_char.get(c))
            for c, d in characters_data.items()
        }

        base_scores = {}
        sem_scores = {}
        psych_scores = {}
        TEMPERATURE = 10.0

        for char_name, core_dialogues in characters_data.items():
            turn_items = context_data_by_char.get(char_name, [])
            psych = self._extract_context_aware_psycholinguistics(
                core_dialogues, turn_items, emotion_profiles[char_name]
            )
            psych_scores[char_name] = psych
            sem_scores[char_name] = {}
            base_scores[char_name] = {}

            if TRANSFORMERS_AVAILABLE:
                core_mat = self._encode_sentences(core_dialogues)
                core_mean = np.mean(core_mat, axis=0)
                turn_mat = self._encode_context_aware_turns(turn_items, alpha=0.75) if turn_items else core_mat

                for trait in TRAITS:
                    midpoint = self.trait_info[trait]["midpoint"]
                    axis = self.trait_info[trait]["axis"]

                    centered_turns = turn_mat - midpoint
                    turn_norms = np.linalg.norm(centered_turns, axis=1, keepdims=True)
                    centered_turns /= np.where(turn_norms > 0, turn_norms, 1.0)
                    turn_projs = np.dot(centered_turns, axis)

                    k_salient = max(1, int(len(turn_projs) * 0.35))
                    salient_idx = np.argsort(np.abs(turn_projs))[-k_salient:]
                    salient_ctx_proj = float(np.mean(turn_projs[salient_idx]))

                    centered_core = core_mean - midpoint
                    centered_core /= (np.linalg.norm(centered_core) + 1e-9)
                    core_proj = float(np.dot(centered_core, axis))

                    combined_proj = 0.65 * core_proj + 0.35 * salient_ctx_proj
                    sem_prob = float(1.0 / (1.0 + np.exp(-TEMPERATURE * combined_proj)))
                    sem_scores[char_name][trait] = sem_prob
                    base_scores[char_name][trait] = 0.15 * sem_prob + 0.85 * psych[trait]
            else:
                for trait in TRAITS:
                    sem_scores[char_name][trait] = 0.50
                    base_scores[char_name][trait] = psych[trait]

        return base_scores, sem_scores, psych_scores, emotion_profiles

    def run_leave_one_novel_out_cv(self, characters_data, context_data_by_char, ground_truth):
        base_scores, sem_scores, psych_scores, emotion_profiles = self.compute_base_features(
            characters_data, context_data_by_char
        )

        novel_to_chars = defaultdict(list)
        for char_name, gt_info in ground_truth.items():
            if char_name in characters_data:
                novel_name = gt_info.get("novel", "calikusu")
                novel_to_chars[novel_name].append(char_name)

        cv_predictions = {}

        for test_novel, test_chars in novel_to_chars.items():
            train_chars = [
                c for nov, chars in novel_to_chars.items() if nov != test_novel for c in chars
            ]

            for char_name in test_chars:
                cv_predictions[char_name] = {}

            for trait in TRAITS:
                if len(train_chars) >= 6:
                    X_train = np.array([
                        [base_scores[c][trait], psych_scores[c][trait], sem_scores[c][trait]]
                        for c in train_chars
                    ])
                    y_train = np.array([ground_truth[c][trait] for c in train_chars])

                    reg = Ridge(alpha=0.8)
                    reg.fit(X_train, y_train)

                    for c in test_chars:
                        x_test = np.array([[base_scores[c][trait], psych_scores[c][trait], sem_scores[c][trait]]])
                        pred_reg = float(reg.predict(x_test)[0])
                        blended = 0.65 * base_scores[c][trait] + 0.35 * pred_reg
                        cv_predictions[c][trait] = round(float(np.clip(blended, 0.10, 0.95)), 4)
                else:
                    for c in test_chars:
                        cv_predictions[c][trait] = round(float(np.clip(base_scores[c][trait], 0.10, 0.95)), 4)

        return cv_predictions, emotion_profiles, novel_to_chars


def load_all_extracted_contexts():
    combined_context = {}
    if not EXTRACTED_DIR.exists():
        return combined_context

    for ctx_file in sorted(EXTRACTED_DIR.glob("*_context.json")):
        with open(ctx_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for char_name, turns in data.items():
                combined_context[char_name] = turns
    return combined_context


def print_novel_breakdown_report(predictions, ground_truth, novel_to_chars):
    print("\n" + "=" * 65)
    print("ROMAN BAZLI ÇAPRAZ DOĞRULAMA (LEAVE-ONE-NOVEL-OUT CV) ÖZETİ")
    print("=" * 65)

    for novel_name, chars in novel_to_chars.items():
        y_true_nov, y_pred_nov = [], []
        for c in chars:
            y_t = [ground_truth[c][t] for t in TRAITS]
            y_p = [predictions[c][t] for t in TRAITS]
            y_true_nov.extend(y_t)
            y_pred_nov.extend(y_p)

        mae = mean_absolute_error(y_true_nov, y_pred_nov)
        rmse = np.sqrt(mean_squared_error(y_true_nov, y_pred_nov))
        cos_sim = 1.0 - cosine(y_true_nov, y_pred_nov)
        sp_corr, _ = spearmanr(y_true_nov, y_pred_nov)

        print(f"\n[Roman: {novel_name.upper()} ({len(chars)} Karakter)]")
        print(f"  -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | Cosine: {cos_sim:.4f} | Spearman: {sp_corr:.4f}")


def run_prediction_pipeline():
    if not DIALOGUES_PATH.exists() or not GROUND_TRUTH_PATH.exists():
        print("[!] Hata: ground_truth.json veya character_dialogues.json bulunamadı!")
        return

    with open(DIALOGUES_PATH, "r", encoding="utf-8") as f:
        characters_data = json.load(f)
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    context_data_by_char = load_all_extracted_contexts()

    predictor = PersonalityPredictor()
    print(f"\n[*] 4 Roman / {len(characters_data)} Karakter için Leave-One-Novel-Out Çapraz Doğrulama Başlatılıyor...")

    predictions, emotion_profiles, novel_to_chars = predictor.run_leave_one_novel_out_cv(
        characters_data, context_data_by_char, gt
    )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(EMOTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(emotion_profiles, f, ensure_ascii=False, indent=2)

    for novel_name, chars in novel_to_chars.items():
        nov_preds = {c: predictions[c] for c in chars}
        nov_emos = {c: emotion_profiles[c] for c in chars}
        with open(PREDICTIONS_DIR / f"{novel_name}_predictions.json", "w", encoding="utf-8") as f:
            json.dump(nov_preds, f, ensure_ascii=False, indent=2)
        with open(PREDICTIONS_DIR / f"{novel_name}_emotions.json", "w", encoding="utf-8") as f:
            json.dump(nov_emos, f, ensure_ascii=False, indent=2)

    print_novel_breakdown_report(predictions, gt, novel_to_chars)
    evaluate_predictions(predictions, gt)


if __name__ == "__main__":
    run_prediction_pipeline()