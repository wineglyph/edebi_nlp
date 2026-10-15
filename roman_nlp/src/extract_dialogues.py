import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"

NOVEL_CONFIGS = {
    "calikusu": {
        "filenames": ["calikusu.txt"],
        "characters": ["Feride", "Kamran", "Hayrullah_Bey", "Mujgan", "Haci_Kalfa", "Munise"],
        "protagonist": "Feride",
        "aliases": {
            "hayrullah bey": "Hayrullah_Bey", "hayrullah": "Hayrullah_Bey",
            "ihtiyar doktor": "Hayrullah_Bey", "doktor bey": "Hayrullah_Bey", "doktor": "Hayrullah_Bey",
            "hacı kalfa": "Haci_Kalfa", "haci kalfa": "Haci_Kalfa", "gülmisal kalfa": "Haci_Kalfa"
        }
    },
    "kurk_mantolu_madonna": {
        "filenames": ["kurk_mantolu_madonna.txt"],
        "characters": ["Raif_Efendi", "Maria_Puder", "Rasim", "Hamdi"],
        "protagonist": "Raif_Efendi",
        "aliases": {
            "raif efendi": "Raif_Efendi", "raif bey": "Raif_Efendi", "herr raif": "Raif_Efendi",
            "maria puder": "Maria_Puder", "maria": "Maria_Puder",
            "hamdi bey": "Hamdi", "hamdi": "Hamdi",
            "rasim": "Rasim"
        }
    },
    "icimizdeki_seytan": {
        "filenames": ["icimizdeki_seytan.txt"],
        "characters": ["Omer", "Macide", "Bedri", "Nihat", "Hafiz_Husamettin"],
        "protagonist": None,
        "aliases": {
            "ömer": "Omer", "omer": "Omer",
            "macide": "Macide",
            "bedri": "Bedri",
            "nihat": "Nihat",
            "hafız hüsamettin": "Hafiz_Husamettin", "hüsamettin efendi": "Hafiz_Husamettin",
            "hafız efendi": "Hafiz_Husamettin", "hafız": "Hafiz_Husamettin", "veznedar": "Hafiz_Husamettin"
        }
    },
    "yaprak_dokumu": {
        "filenames": ["yaprak_dokumu.txt", "380862681-Resat-Nuri-Gu-ntekin-Yaprak-Do-ku-mu-pdf.txt"],
        "characters": ["Ali_Riza_Bey", "Sevket", "Fikret", "Ferhunde", "Leyla"],
        "protagonist": None,
        "aliases": {
            "ali rıza bey": "Ali_Riza_Bey", "ali rıza": "AliRiza_Bey",
            "şevket": "Sevket", "sevket": "Sevket",
            "fikret": "Fikret",
            "ferhunde": "Ferhunde", "gelin": "Ferhunde", "yengeleri": "Ferhunde",
            "şevket'in karısı": "Ferhunde", "genç kadın": "Ferhunde",
            "leyla": "Leyla"
        }
    }
}

FIRST_PERSON_VERBS = (
    r"\b(?:dedim|diyordum|söyledim|sordum|cevap verdim|karşılık verdim|"
    r"bağırdım|haykırdım|mırıldandım|fısıldadım|ekledim|ilave ettim|"
    r"atıldım|yalvardım|sızlandım|itiraf ettim|tekrar ettim)\b"
)

THIRD_PERSON_VERBS = (
    r"\b(?:dedi|diyordu|söyledi|söyleniyordu|sordu|soruyordu|cevap verdi|"
    r"karşılık verdi|bağırdı|bağırıyordu|haykırdı|mırıldandı|fısıldadı|"
    r"ekledi|ilave etti|atıldı|seslendi|inledi|sızlandı|itiraz etti|"
    r"diye\s+[a-zçğıöşüâîû]+\s*[a-zçğıöşüâîû]*)\b"
)

ALL_SPEECH_VERBS = rf"(?:{FIRST_PERSON_VERBS}|{THIRD_PERSON_VERBS})"
OBLIQUE_CASE_SUFFIX = r"^['’]?(?:[yhn]?[ıiuü](?:n|z)?|[y]?[ae]|d[ae]n?|l[ae]|n[ıiuü]n)\b"


class MultiNovelDialogueExtractor:
    def __init__(self, novel_id, config):
        self.novel_id = novel_id
        self.config = config
        self.target_characters = config["characters"]
        self.protagonist = config.get("protagonist")
        self.alias_map = {}
        self._build_aliases()

    @staticmethod
    def _normalize_tr(text):
        replacements = str.maketrans("âîûÂÎÛçğıöşüÇĞİÖŞÜ", "aiuAIUcgiosuCGIOSU")
        return text.translate(replacements).lower()

    def _build_aliases(self):
        for canonical in self.target_characters:
            clean = canonical.replace("_", " ").lower()
            first = clean.split()[0]
            self.alias_map[clean] = canonical
            self.alias_map[first] = canonical
            self.alias_map[self._normalize_tr(clean)] = canonical
            self.alias_map[self._normalize_tr(first)] = canonical

        for alias, canonical in self.config.get("aliases", {}).items():
            self.alias_map[alias.lower()] = canonical
            self.alias_map[self._normalize_tr(alias)] = canonical

    def _preprocess_novel_text(self, raw_text):
        # 1. Yaprak Dökümü OCR (Đ -> İ) ve Romen rakamlı bölüm başlığı (- XİV -) temizliği
        text = raw_text.replace("Đ", "İ").replace("đ", "i").replace("\f", "\n")
        text = re.sub(r"^\s*-\s*[IVXİıL]+\s*-\s*$", "", text, flags=re.MULTILINE)

        # 2. Kürk Mantolu Madonna satır sonu (¬) ve satır içi sayfa başlığı temizliği
        text = re.sub(r"¬\s*\n\s*", "", text)
        text = text.replace("¬", "")
        text = re.sub(r"«([^»\n]{5,300})\*", r"«\1»", text)
        text = re.sub(r"\b\d*\s*KÜRK\s*MANTOLU\s*MADONNA\s*\d*\b", " ", text, flags=re.IGNORECASE)

        # 3. İçimizdeki Şeytan önsöz kesimi
        if self.novel_id == "icimizdeki_seytan":
            m_start = re.search(r"İÇİMİZDEKİ ŞEYTAN\s*\n+\s*[1lI]\b", text)
            if m_start:
                text = text[m_start.start():]

        # 4. Çalıkuşu sözlük ve mükerrer metin kesimi
        if self.novel_id == "calikusu":
            glossary_match = re.search(r"\n\s*Bibliothe[qç]ue Rose\s*:", text)
            if glossary_match:
                text = text[:glossary_match.start()]
            half_len = len(text) // 2
            sample_chunk = text[1000:2500]
            if sample_chunk and text.find(sample_chunk, half_len - 5000) != -1:
                dup_idx = text.find(sample_chunk, half_len - 5000) - 1000
                text = text[:max(dup_idx, half_len)]

        header_pat = re.compile(
            r"^\s*(?:ÇALIKUŞU.*|KÜRK MANTOLU MADONNA.*|İçimizdeki Şeytan\s*\d*|"
            r"\d+\s+Sabahattin Ali|Reşat Nuri Güntekin.*|www\.kitapsevenler\.com|"
            r"-?\s*[IVXİıL]+\s*-?|\d+)\s*$",
            re.IGNORECASE
        )

        cleaned_lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s or header_pat.match(s):
                continue
            s = re.sub(r"^\s*-\s*[IVXİıL]+\s*-\s*", "", s)
            if s:
                cleaned_lines.append(s)

        paragraphs = []
        current = ""
        for line in cleaned_lines:
            is_new_dlg = bool(re.match(r"^[-—–«“\"]", line))
            if not current:
                current = line
            elif is_new_dlg:
                paragraphs.append(current)
                current = line
            else:
                if current.endswith("-") and line and line[0].islower():
                    current = current[:-1] + line
                elif not re.search(r'[.!?:"”»]\s*$', current):
                    current = f"{current} {line}"
                else:
                    paragraphs.append(current)
                    current = line
        if current:
            paragraphs.append(current)

        return paragraphs

    def _split_utterance_and_narrator(self, raw_dialogue):
        text = re.sub(r"^[-—–\s]+", "", raw_dialogue).strip()
        m = re.search(
            rf"(?:,\s*|[.!?;]\s+)(?:[a-zçğıöşüâîû\s,]{{0,30}}?\b)?({ALL_SPEECH_VERBS}.*)$",
            text,
            flags=re.IGNORECASE
        )
        if m and m.start() > 12:
            return text[:m.start()].strip(), text[m.start():].strip(" ,.;")
        return text, ""

    def _is_addressed_in_utterance(self, canonical_char, utterance):
        utt_lower = utterance.lower()
        utt_norm = self._normalize_tr(utterance)
        for alias, mapped_char in self.alias_map.items():
            if mapped_char != canonical_char:
                continue
            for pat in (rf"(?:^|[,!?;.]\s*){re.escape(alias)}\b", rf"\b{re.escape(alias)}\s*[,!?.]"):
                if re.search(pat, utt_lower) or re.search(pat, utt_norm):
                    return True
        return False

    def _has_nominative_mention(self, alias, text_lower, text_norm):
        for target_text in (text_lower, text_norm):
            for m in re.finditer(rf"\b{re.escape(alias)}\b", target_text):
                rem = target_text[m.end():]
                if not re.match(OBLIQUE_CASE_SUFFIX, rem):
                    return True
        return False

    def _resolve_speaker(self, clean_utt, inline_narr, prev_para, active_protagonist, in_raif_diary=True):
        if clean_utt.endswith(":"):
            return None

# 2. _resolve_speaker fonksiyonunun hemen başına (if clean_utt.endswith(":"): altına) şu filtreyi ekle:
        # Raif Efendi hakkında memurların söylediği alaycı cümleleri filtrele
        if self.novel_id == "kurk_mantolu_madonna" and any(
            bad in clean_utt.lower() for bad in ["hımbılın biridir", "mızmız, hastalıklı haline"]
        ):
            return None

        # Yaprak Dökümü'nde Ferhunde'nin bulunduğu sahne paragraflarındaki konuşmaları yakala
        if self.novel_id == "yaprak_dokumu":
            scene_ctx = f"{inline_narr} {prev_para}".lower()
            if "ferhunde" in scene_ctx and not any(k in scene_ctx for k in ["ali rıza", "fikret", "leyla"]):
                return "Ferhunde"
        # Kürk Mantolu Madonna'da günlük başlamadan önce Maria_Puder henüz sahnede yoktur
        utt_lower = clean_utt.lower()
        inline_lower = inline_narr.lower()
        prev_tail = prev_para[-130:].lower() if prev_para else ""

        if active_protagonist and re.search(FIRST_PERSON_VERBS, inline_lower):
            if not self._is_addressed_in_utterance(active_protagonist, clean_utt):
                return active_protagonist

        if active_protagonist and prev_tail.endswith(":") and re.search(FIRST_PERSON_VERBS, prev_tail):
            if not self._is_addressed_in_utterance(active_protagonist, clean_utt):
                return active_protagonist

        if re.search(FIRST_PERSON_VERBS, inline_lower) or re.search(FIRST_PERSON_VERBS, utt_lower):
            return active_protagonist

        local_window = f"{inline_lower} {prev_tail}"
        local_norm = self._normalize_tr(local_window)

        for alias, canonical in sorted(self.alias_map.items(), key=lambda x: -len(x[0])):
            if self.novel_id == "kurk_mantolu_madonna" and not in_raif_diary and canonical == "Maria_Puder":
                continue
            if self._is_addressed_in_utterance(canonical, clean_utt):
                continue
            if self._has_nominative_mention(alias, local_window, local_norm):
                return canonical

        return None

    def extract(self, raw_text):
        paragraphs = self._preprocess_novel_text(raw_text)
        dialogues_only = defaultdict(list)
        dialogues_with_context = defaultdict(list)
        seen_utterances = defaultdict(set)

        quote_pattern = re.compile(r'«([^»]+)»|“([^”]+)”|"([^"]+)"')
        # Yaprak Dökümü gibi tırnaksız iç konuşma/bağırma kalıbı: "... : <Cümle> diye haykırıyordu/diyordu"
        unquoted_speech_pat = re.compile(
            r"(?:[:]|çarp(?:arak|ıp))\s*([A-ZÇĞİÖŞÜ][^a-zçğıöşü]{0,2}[a-zçğıöşü].{20,260}?)\s+diye\s+(?:haykır|bağır|söylen|feryat)",
            re.UNICODE
        )
        in_raif_diary = False

        for idx, para in enumerate(paragraphs):
            if self.novel_id == "kurk_mantolu_madonna":
                if "20 haziran 1933" in para.lower():
                    in_raif_diary = True
                active_protagonist = "Raif_Efendi" if in_raif_diary else "Rasim"
            else:
                active_protagonist = self.protagonist

            prev_para = paragraphs[idx - 1] if idx > 0 else ""
            next_para = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else ""

            is_dash = bool(re.match(r"^[-—–]\s*", para))
            candidates = []

            if is_dash:
                clean_utt, inline_narr = self._split_utterance_and_narrator(para)
                candidates.append((clean_utt, inline_narr))
            else:
                q_matches = list(quote_pattern.finditer(para))
                if q_matches:
                    for qm in q_matches:
                        raw_q = (qm.group(1) or qm.group(2) or qm.group(3) or "").strip()
                        if len(raw_q) < 18:
                            continue
                        clean_utt, inline_narr = self._split_utterance_and_narrator(raw_q)
                        outer_narr = (para[:qm.start()] + " " + para[qm.end():]).strip()
                        candidates.append((clean_utt, f"{inline_narr} {outer_narr}".strip()))
                elif self.novel_id == "yaprak_dokumu":
                    for um in unquoted_speech_pat.finditer(para):
                        clean_utt = um.group(1).strip()
                        outer_narr = (para[:um.start(1)] + " " + para[um.end(1):]).strip()
                        candidates.append((clean_utt, outer_narr))

            for clean_utt, inline_narr in candidates:
                if len(clean_utt) < 18 or len(clean_utt.split()) < 4:
                    continue

                speaker = self._resolve_speaker(
                    clean_utt, inline_narr, prev_para, active_protagonist, in_raif_diary
                )
                if not speaker:
                    continue

                norm_key = re.sub(r"\W+", "", clean_utt.lower())
                if norm_key in seen_utterances[speaker]:
                    continue
                seen_utterances[speaker].add(norm_key)

                ctx_parts = []
                if inline_narr:
                    ctx_parts.append(inline_narr)
                if prev_para and not prev_para.startswith("-"):
                    ctx_parts.append(prev_para)
                if next_para and not next_para.startswith("-"):
                    ctx_parts.append(next_para)

                clean_ctx = re.sub(r"\s+", " ", " ".join(ctx_parts)).strip()[:320]
                dialogues_only[speaker].append(clean_utt)
                dialogues_with_context[speaker].append({
                    "utterance": clean_utt,
                    "narrator_context": clean_ctx
                })

        return dict(dialogues_only), dict(dialogues_with_context)


def find_novel_file(filenames):
    search_dirs = [
        BASE_DIR / "data" / "novels",
        BASE_DIR,
        BASE_DIR.parent,
        BASE_DIR.parent / "dataset" / "calikusu",
        BASE_DIR.parent / "dataset" / "kurk_mantolu_madonna",
        BASE_DIR.parent / "dataset" / "icimizdeki_seytan",
        BASE_DIR.parent / "dataset" / "yaprak_dokumu"
    ]
    for d in search_dirs:
        for fname in filenames:
            candidate = d / fname
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Çoklu Roman Otomatik Diyalog ve Bağlam Çıkarıcı v4.1")
    parser.add_argument("--novel", type=str, default="all", help="İşlenecek roman adı veya 'all'")
    args = parser.parse_args()

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    target_novels = NOVEL_CONFIGS.keys() if args.novel == "all" else [args.novel]

    print("=" * 65)
    print("ÇOKLU ROMAN DİYALOG VE BAĞLAM PENCERESİ AYRIŞTIRICI (v4.1)")
    print("=" * 65)

    for novel_id in target_novels:
        if novel_id not in NOVEL_CONFIGS:
            continue
        cfg = NOVEL_CONFIGS[novel_id]
        novel_path = find_novel_file(cfg["filenames"])

        if not novel_path:
            print(f"\n[!] '{novel_id}' metin dosyası bulunamadı ({cfg['filenames'][0]}). Atlanıyor...")
            continue

        with open(novel_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        extractor = MultiNovelDialogueExtractor(novel_id, cfg)
        dialogues, context_data = extractor.extract(raw_text)

        dlg_out = EXTRACTED_DIR / f"{novel_id}_dialogues.json"
        ctx_out = EXTRACTED_DIR / f"{novel_id}_context.json"

        with open(dlg_out, "w", encoding="utf-8") as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=2)
        with open(ctx_out, "w", encoding="utf-8") as f:
            json.dump(context_data, f, ensure_ascii=False, indent=2)

        total_utts = sum(len(v) for v in dialogues.values())
        print(f"\n[*] Roman: {novel_id.upper()} (Kaynak: {novel_path.name})")
        for char_name in cfg["characters"]:
            cnt = len(dialogues.get(char_name, []))
            print(f"    -> {char_name:18s}: {cnt:3d} replik (+ anlatıcı bağlamı)")
        print(f"    [+] Kaydedildi: data/extracted/{dlg_out.name} ({total_utts} toplam replik)")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()