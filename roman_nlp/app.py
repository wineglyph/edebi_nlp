import json
import re
import sys
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- DİZİN VE DOSYA YOLLARI ---
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

GROUND_TRUTH_PATH = BASE_DIR / "data" / "ground_truth.json"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.json"
EMOTIONS_PATH = BASE_DIR / "data" / "emotion_profiles.json"
DIALOGUES_PATH = BASE_DIR / "data" / "character_dialogues.json"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"

st.set_page_config(
    page_title="Türk Romanında Karakter ve Kişilik Analizi",
    layout="wide",
    initial_sidebar_state="auto",
)

def render_html(html_str):
    """
    HTML bloklarının Streamlit Markdown tarafından <pre><code> kod bloğu
    olarak algılanmasını önlemek için soldaki boşlukları temizleyerek basar.
    """
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# --- EDEBİYAT DOSTU AÇIKLAMALAR VE METADATA ---
NOVELS = {
    "calikusu": {
        "title": "Çalıkuşu",
        "author": "Reşat Nuri Güntekin",
        "year": "1922",
        "summary": "İstanbul'dan Anadolu'ya uzanan bir genç kızın idealizm, yalnızlık ve irade mücadelesi."
    },
    "kurk_mantolu_madonna": {
        "title": "Kürk Mantolu Madonna",
        "author": "Sabahattin Ali",
        "year": "1943",
        "summary": "İçe kapanık bir ruhun sanat, yabancılaşma ve unutulmaz bir aşk etrafında şekillenen iç dünyası."
    },
    "icimizdeki_seytan": {
        "title": "İçimizdeki Şeytan",
        "author": "Sabahattin Ali",
        "year": "1940",
        "summary": "İrade zayıflığı, sorumluluktan kaçış ve vicdan muhasebesi üzerine derin bir karakter romanı."
    },
    "yaprak_dokumu": {
        "title": "Yaprak Dökümü",
        "author": "Reşat Nuri Güntekin",
        "year": "1930",
        "summary": "Ahlaki ilkelerine bağlı bir babanın, değişen toplum ve savrulan çocukları karşısındaki trajedisi."
    }
}

CHARACTERS_INFO = {
    "Feride": ("Çalıkuşu", "Başkahraman / Muallime", "Hayat dolu, özgürlüğüne düşkün, haksızlığa boyun eğmeyen; içindeki derin acıyı neşesi ve cesaretiyle örten güçlü bir genç kadın."),
    "Kamran": ("Çalıkuşu", "Feride'nin Kuzeni ve Nişanlısı", "Kibar, duygusal ve uyumlu; ancak gençliğinde kararsızlığı yüzünden hata yapıp sonrasında derin bir pişmanlık yaşayan karakter."),
    "Hayrullah_Bey": ("Çalıkuşu", "Askerî Doktor", "Dobra, disiplini yüksek, soğukkanlı ve son derece koruyucu; Feride'ye babalık yapan bilge doktor."),
    "Mujgan": ("Çalıkuşu", "Feride'nin Kuzeni ve Sırdaşı", "Sakin, güvenilir, sır saklayan ve olaylar karşısında sağduyusunu koruyan şefkatli abla figürü."),
    "Haci_Kalfa": ("Çalıkuşu", "Otel Kalfası", "Konuşkan, yardımsever, geleneklere bağlı ve pratik zekâsıyla Feride'ye kol kanat geren sıcakkanlı karakter."),
    "Munise": ("Çalıkuşu", "Feride'nin Manevi Kızı", "Sevgiye muhtaç, terk edilmekten korkan, uyumlu ve duygusal açıdan çok kırılgan küçük kız."),
    "Raif_Efendi": ("Kürk Mantolu Madonna", "Almanca Mütercimi", "Dış dünyaya karşı aşırı sessiz ve çekingen; ancak sanat, edebiyat ve sevgi konusunda çok derin bir iç dünyaya sahip melankolik karakter."),
    "Maria_Puder": ("Kürk Mantolu Madonna", "Ressam", "Özgür ruhlu, açık sözlü, kimseden merhamet beklemeyen, kendi ayakları üzerinde duran güçlü ve sıra dışı bir sanatçı."),
    "Rasim": ("Kürk Mantolu Madonna", "Anlatıcı / Genç Memur", "İnsan ruhunu anlamaya meraklı, vefalı, saygılı ve Raif Efendi'nin iç dünyasına tek tanık olan duyarlı gözlemci."),
    "Hamdi": ("Kürk Mantolu Madonna", "Şirket Şefi", "Edebiyatı ve hayalleri boş gören, sadece pratik başarıya ve otoriteye önem veren baskın karakter."),
    "Omer": ("İçimizdeki Şeytan", "Üniversiteli Genç", "Hayal gücü ve felsefi sorgulamaları güçlü; ancak iradesizliği ve sorumluluktan kaçışı yüzünden hatalarını 'içindeki şeytana' yükleyen kararsız ruh."),
    "Macide": ("İçimizdeki Şeytan", "Konservatuvar Öğrencisi", "Müziğe ve sanata bağlı, dürüst, tertipli, iradeli ve en zor anlarda bile soğukkanlılığını koruyan güçlü kadın karakter."),
    "Bedri": ("İçimizdeki Şeytan", "Müzik Öğretmeni", "Sanatkâr ruhlu, ailesine ve dostlarına karşı son derece sorumluluk sahibi, karşılık beklemeden fedakârlık yapan olgun karakter."),
    "Nihat": ("İçimizdeki Şeytan", "Ömer'in Çevresinden İdeolog", "Merhameti zayıflık sayan, güce tapan, insanları kendi çıkarı için yönlendiren baskın ve çatışmacı karakter."),
    "Hafiz_Husamettin": ("İçimizdeki Şeytan", "Veznedar", "Yıllarca dürüst çalıştıktan sonra ailesine yardım etmek isterken kasa açığına düşen ve büyük bir vicdan/korku bunalımı yaşayan memur."),
    "Ali_Riza_Bey": ("Yaprak Dökümü", "Emekli Memur / Aile Babası", "Doğruluk, namus ve ahlak ilkelerine ömrü boyunca sıkı sıkıya bağlı; çocuklarına kıyamayan ama ailenin dağılışını çaresizce izleyen baba."),
    "Sevket": ("Yaprak Dökümü", "Evin Büyük Oğlu", "Babası gibi dürüst ve vazifeşinas; bütün ailenin geçim yükünü tek başına sırtlanan, ancak eşinin savurganlığı yüzünden felakete sürüklenen evlat."),
    "Fikret": ("Yaprak Dökümü", "Evin Büyük Kızı", "Ailenin nereye sürüklendiğini en baştan gören, tutumlu, ilkeli ama kardeşlerinin sorumsuzluğu karşısında sert ve hırçınlaşan karakter."),
    "Ferhunde": ("Yaprak Dökümü", "Evin Gelini", "Eğlenceye, gösterişe ve rahata düşkün; sorumluluktan kaçan, evdeki düzeni kendi isteğine göre yönetmeye çalışan baskın karakter."),
    "Leyla": ("Yaprak Dökümü", "Ortanca Kız Kardeş", "Sosyete hayatına ve lükse özenen, iradesi zayıf, hayal kırıklıkları sonrasında derin ruhsal çöküntüler yaşayan kırılgan karakter.")
}

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

TRAIT_INFO = {
    "openness": ("Yeniliğe Açıklık", "Hayal gücü, sanat/edebiyat merakı ve farklı düşüncelere açıklık"),
    "conscientiousness": ("Sorumluluk Bilinci", "Görev duygusu, irade gücü, düzenlilik ve ilkeli davranış"),
    "extraversion": ("Dışadönüklük", "Sosyal enerji, konuşkanlık, girişkenlik ve duyguları dışa vurma"),
    "agreeableness": ("Uyumluluk ve Şefkat", "Merhamet, fedakârlık, nezaket ve insanlarla iyi geçinme"),
    "neuroticism": ("Duygusal Hassasiyet", "Kaygı, hüzün, stres karşısında kırılganlık ve içsel fırtınalar")
}

EMOTION_INFO = {
    "trust_affection": ("Şefkat ve Güven", "#3B6E5C"),
    "joy": ("Neşe ve Coşku", "#D99B26"),
    "curiosity": ("Merak ve Sorgulama", "#3A7CA5"),
    "sadness": ("Hüzün ve Keder", "#6D7278"),
    "fear": ("Kaygı ve Korku", "#8E5572"),
    "anger": ("Öfke ve Çatışma", "#B84A39")
}

EMOTION_INFO_DARK = {
    "trust_affection": ("Şefkat ve Güven", "#5BA88C"),
    "joy": ("Neşe ve Coşku", "#E6A838"),
    "curiosity": ("Merak ve Sorgulama", "#4FA3D1"),
    "sadness": ("Hüzün ve Keder", "#9AA0A6"),
    "fear": ("Kaygı ve Korku", "#B87B9C"),
    "anger": ("Öfke ve Çatışma", "#E06A57")
}

# --- KULLANICI DOSTU, SADE VE ŞIK CSS TASARIMI (AÇIK / KOYU TEMA DESTEKLİ) ---
def inject_clean_literary_css(is_dark=False):
    if is_dark:
        bg_app = "#141312"
        text_primary = "#EDE8DF"
        text_secondary = "#A8A298"
        text_tertiary = "#8C867D"
        card_bg = "#1D1C19"
        card_border = "#332F2A"
        card_hover_border = "#4A443C"
        accent_color = "#E07A5F"
        accent_subtle = "rgba(224, 122, 95, 0.12)"
        accent_border = "rgba(224, 122, 95, 0.25)"
        bar_bg = "#2B2824"
        bar_fill = "linear-gradient(90deg, #E07A5F, #C85A3C)"
        dialogue_bg = "#181715"
        dialogue_border = "#2E2B26"
        tab_list_bg = "#262420"
        tab_selected_bg = "#141312"
        input_bg = "#1D1C19"
        input_border = "#38342F"
        dropdown_hover_bg = "#2E2A25"
        sidebar_bg = "#181715"
        sidebar_border = "#302C26"
        box_shadow_header = "0 4px 24px rgba(0, 0, 0, 0.45)"
        box_shadow_card = "0 2px 14px rgba(0, 0, 0, 0.3)"
    else:
        bg_app = "#FAF7F2"
        text_primary = "#1F1E1B"
        text_secondary = "#5C5852"
        text_tertiary = "#78736A"
        card_bg = "#FFFFFF"
        card_border = "#E6E0D5"
        card_hover_border = "#D8D0C5"
        accent_color = "#9E3B24"
        accent_subtle = "#FDF3F0"
        accent_border = "#F5D4CB"
        bar_bg = "#EFECE6"
        bar_fill = "linear-gradient(90deg, #C86D51, #9E3B24)"
        dialogue_bg = "#FCFBF9"
        dialogue_border = "#EAE4D9"
        tab_list_bg = "#EAE4D9"
        tab_selected_bg = "#FFFFFF"
        input_bg = "#FFFFFF"
        input_border = "#D8D0C5"
        dropdown_hover_bg = "#FAF2EF"
        sidebar_bg = "#F7F3EC"
        sidebar_border = "#E6E0D5"
        box_shadow_header = "0 4px 20px rgba(31, 30, 27, 0.03)"
        box_shadow_card = "0 2px 12px rgba(31, 30, 27, 0.025)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,500;0,600;1,400&display=swap');

    /* Streamlit gereksiz üst barları gizle */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Streamlit üst başlık çubuğunun tıklamaları engellemesini önle */
    header[data-testid="stHeader"],
    .stAppHeader {{
        background: transparent !important;
        pointer-events: none !important;
    }}

    header[data-testid="stHeader"] button,
    .stAppHeader button,
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg {{
        color: {text_primary} !important;
        pointer-events: auto !important;
    }}

    .stApp {{
        background-color: {bg_app} !important;
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif;
    }}

    .block-container {{
        max-width: 1180px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 4rem !important;
    }}

    @keyframes fadeInSmooth {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes fillBarSmooth {{
        from {{ width: 0%; }}
    }}

    /* Üst Başlık Alanı */
    .site-header {{
        text-align: center;
        padding: 2.2rem 1.5rem 1.8rem 1.5rem;
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: {box_shadow_header} !important;
        animation: fadeInSmooth 0.5s ease-out;
    }}

    .site-badge {{
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {accent_color} !important;
        background: {accent_subtle} !important;
        border: 1px solid {accent_border} !important;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 0.8rem;
    }}

    .site-title {{
        font-family: 'Lora', Georgia, serif;
        font-size: 2.35rem;
        font-weight: 600;
        color: {text_primary} !important;
        margin: 0 0 0.6rem 0;
        line-height: 1.2;
    }}

    .site-desc {{
        font-size: 1rem;
        color: {text_secondary} !important;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    /* İçerik Kartları */
    .clean-card {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: {box_shadow_card} !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        animation: fadeInSmooth 0.45s ease-out;
    }}

    .clean-card:hover {{
        box-shadow: 0 6px 22px rgba(0, 0, 0, {0.45 if is_dark else 0.055}) !important;
        border-color: {card_hover_border} !important;
    }}

    .card-tag {{
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {accent_color} !important;
        margin-bottom: 0.3rem;
    }}

    .card-title {{
        font-family: 'Lora', Georgia, serif;
        font-size: 1.65rem;
        font-weight: 600;
        color: {text_primary} !important;
        margin: 0 0 0.5rem 0;
    }}

    .card-subtitle {{
        font-size: 0.9rem;
        font-weight: 500;
        color: {text_secondary} !important;
        margin-bottom: 0.9rem;
    }}

    .card-body {{
        font-size: 0.96rem;
        line-height: 1.65;
        color: {text_primary} !important;
    }}

    /* Kişilik Çubukları */
    .trait-item {{
        margin-bottom: 1.15rem;
    }}

    .trait-top {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.35rem;
    }}

    .trait-name {{
        font-weight: 600;
        font-size: 0.95rem;
        color: {text_primary} !important;
    }}

    .trait-score {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {accent_color} !important;
    }}

    .trait-bar-bg {{
        width: 100%;
        height: 10px;
        background: {bar_bg} !important;
        border-radius: 6px;
        overflow: hidden;
    }}

    .trait-bar-fill {{
        height: 100%;
        background: {bar_fill} !important;
        border-radius: 6px;
        animation: fillBarSmooth 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}

    .trait-hint {{
        font-size: 0.8rem;
        color: {text_tertiary} !important;
        margin-top: 0.25rem;
    }}

    /* Romandan Alıntı Kartları */
    .dialogue-box {{
        background: {dialogue_bg} !important;
        border-left: 4px solid {accent_color} !important;
        border-top: 1px solid {dialogue_border} !important;
        border-right: 1px solid {dialogue_border} !important;
        border-bottom: 1px solid {dialogue_border} !important;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.85rem;
        transition: transform 0.2s ease, background 0.2s ease;
    }}

    .dialogue-box:hover {{
        transform: translateX(4px);
        background: {card_bg} !important;
    }}

    .dialogue-quote {{
        font-family: 'Lora', Georgia, serif;
        font-style: italic;
        font-size: 1.05rem;
        line-height: 1.6;
        color: {text_primary} !important;
    }}

    .dialogue-context {{
        font-size: 0.82rem;
        color: {text_tertiary} !important;
        margin-top: 0.45rem;
        padding-top: 0.45rem;
        border-top: 1px dashed {card_border} !important;
    }}

    /* Edebi İkiz Detay Kutuları */
    .twin-reason-box {{
        background: {dialogue_bg} !important;
        border: 1px solid {card_border} !important;
        color: {text_primary} !important;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }}

    .twin-sub-card {{
        color: {text_primary} !important;
        font-size: 0.92rem;
        margin-top: 0.4rem;
    }}

    .twin-sub-hint {{
        color: {text_secondary} !important;
        font-size: 0.84rem;
    }}

    .twin-sub-divider {{
        border: none !important;
        border-top: 1px solid {card_border} !important;
        margin: 0.6rem 0;
    }}

    /* Genel Metin Elemanları */
    p, span, label, div {{
        color: {text_primary};
    }}

    /* Widget Etiketleri (Selectbox, Textarea, Radio Üst Başlıkları) */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label {{
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }}

    /* Radio Buton Seçenek Yazıları ve Kutusu */
    .stRadio,
    .stRadio *,
    .stRadio div[role="radiogroup"],
    .stRadio label,
    .stRadio label p,
    .stRadio label span,
    .stRadio label div {{
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }}

    [data-testid="stRadioOption"] {{
        cursor: pointer !important;
        padding: 4px 8px !important;
        border-radius: 8px !important;
        transition: background 0.15s ease !important;
    }}

    [data-testid="stRadioOption"]:hover {{
        background: {accent_subtle} !important;
    }}

    /* Radio Buton Çemberleri */
    .stRadio [data-baseweb="radio"] div:first-child,
    .stRadio div[data-rac] div:first-child {{
        border-color: {accent_color} !important;
    }}

    /* Sekme (Tab) Görünümü */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.6rem;
        background: {tab_list_bg} !important;
        padding: 0.4rem;
        border-radius: 10px;
        border: 1px solid {card_border} !important;
        margin-bottom: 1.5rem;
    }}

    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: {text_secondary} !important;
        border-radius: 8px;
        border: none !important;
        background: transparent !important;
        opacity: 1 !important;
    }}

    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover *,
    .stTabs [data-baseweb="tab"]:hover p {{
        color: {accent_color} !important;
    }}

    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {{
        background: {tab_selected_bg} !important;
        color: {accent_color} !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, {0.35 if is_dark else 0.08}) !important;
    }}

    /* Seçim Kutuları (Selectbox: React-Aria ve BaseWeb Desteği) */
    [data-testid="stSelectbox"] div[data-rac][role="group"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        color: {text_primary} !important;
        border-radius: 8px !important;
    }}

    [data-testid="stSelectbox"] input,
    [data-testid="stSelectbox"] [role="combobox"] {{
        background-color: transparent !important;
        border: none !important;
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.94rem !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }}

    [data-testid="stSelectbox"] button {{
        background-color: transparent !important;
        border: none !important;
        color: {text_primary} !important;
    }}

    [data-testid="stSelectbox"] svg {{
        fill: {text_primary} !important;
        color: {text_primary} !important;
    }}

    /* Metin Alanları (Textarea) ve Standart Inputlar */
    textarea,
    input:not([role="combobox"]):not([type="radio"]) {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        color: {text_primary} !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }}

    textarea:focus,
    input:focus,
    [data-testid="stSelectbox"] div[data-rac][role="group"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {accent_color} !important;
        box-shadow: 0 0 0 1px {accent_color} !important;
    }}

    textarea::placeholder,
    input::placeholder {{
        color: {text_tertiary} !important;
        opacity: 1 !important;
    }}

    /* Açılır Liste (Dropdown Popover & Options - React-Aria ve BaseWeb) */
    .react-aria-Popover,
    div[data-rac][role="dialog"],
    div[role="listbox"],
    ul[role="listbox"],
    [data-testid="stSelectboxVirtualDropdown"] {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 28px rgba(0, 0, 0, {0.55 if is_dark else 0.12}) !important;
        overflow: hidden !important;
    }}

    div[role="option"],
    div[role="option"] *,
    [data-rac][role="option"],
    [data-rac][role="option"] *,
    [role="option"],
    [role="option"] *,
    ul[role="listbox"] li,
    ul[role="listbox"] li * {{
        background-color: {input_bg} !important;
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
    }}

    div[role="option"]:hover,
    div[role="option"]:hover *,
    div[role="option"][data-focused="true"],
    div[role="option"][data-focused="true"] *,
    div[role="option"][data-selected="true"],
    div[role="option"][data-selected="true"] *,
    [role="option"]:hover,
    [role="option"]:hover *,
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li:hover * {{
        background-color: {dropdown_hover_bg} !important;
        color: {accent_color} !important;
    }}

    /* Expander / Akordeon */
    [data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {{
        color: {text_primary} !important;
    }}

    /* Birincil Buton (Primary Button) */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {{
        background-color: {accent_color} !important;
        color: {"#141312" if is_dark else "#FFFFFF"} !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
    }}
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {{
        opacity: 0.9 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }}

    /* Yan Menü (Sidebar) */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
    }}

    /* Tablo Alanı (DataFrame) */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {card_border} !important;
        border-radius: 8px;
        overflow: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- VERİ YÜKLEME ---
@st.cache_data
def load_data():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(EMOTIONS_PATH, "r", encoding="utf-8") as f:
        emotions = json.load(f)
    with open(DIALOGUES_PATH, "r", encoding="utf-8") as f:
        core_dialogues = json.load(f)

    extracted_contexts = {}
    if EXTRACTED_DIR.exists():
        for p in sorted(EXTRACTED_DIR.glob("*_context.json")):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for c_name, turns in data.items():
                    extracted_contexts[c_name] = turns

    return gt, preds, emotions, core_dialogues, extracted_contexts

# --- EDEBİ İKİZ İÇİN KISA METİN & GÜNLÜK DİL ANALİZÖRÜ ---
def analyze_short_user_text(text):
    """
    Kullanıcının yazdığı 1-5 cümlelik kısa metinleri hem edebi hem de günlük Türkçe
    ipuçlarıyla psikometrik [0.12, 0.94] aralığında dengeli puanlar.
    """
    t_low = text.lower()

    signals = {
        "openness": {
            "pos": [
                r"kitap", r"sanat", r"resim", r"müzik", r"şiir", r"edebiyat", r"hayal",
                r"felsef", r"merak", r"keşf", r"yeni", r"farklı", r"özgür", r"okum",
                r"düşün", r"anlam", r"hakikat", r"piyano", r"tiyatro", r"gez", r"ufuk",
                r"bağımsız", r"ideal", r"öğren", r"kültür", r"derinlik"
            ],
            "neg": [
                r"boş iş", r"ne gerek", r"adet", r"gelenek", r"eski usul", r"icat çıkarma",
                r"sıradan", r"alışkanlık", r"sadece para", r"ticaret"
            ]
        },
        "conscientiousness": {
            "pos": [
                r"vazife", r"görev", r"sorumlu", r"çalış", r"düzen", r"plan", r"sabah erken",
                r"disiplin", r"sözüm", r"emanet", r"namus", r"doğruluk", r"ilke", r"tutumlu",
                r"idare", r"bitir", r"asla aksat", r"fedakâr", r"emek", r"gayret", r"sabır",
                r"ayaklarımın üzerinde", r"dimdik", r"dürüst", r"irade", r"azim", r"dik dur"
            ],
            "neg": [
                r"üşen", r"sıkıl", r"ertele", r"boşver", r"keyf", r"eğlence", r"para harca",
                r"savurgan", r"iradesiz", r"dayanam", r"süse", r"lüks", r"tembel", r"sorumsuz"
            ]
        },
        "extraversion": {
            "pos": [
                r"kalabalık", r"arkadaş", r"sohbet", r"konuş", r"eğlen", r"neşe", r"gül",
                r"dans", r"sosyete", r"cesur", r"atıl", r"enerji", r"coşku", r"haykır",
                r"part", r"gez", r"insan içine", r"saklamam", r"apaçık", r"yaşamaya değer",
                r"canlı", r"hareketli", r"kahkaha", r"lafımı esirgemem"
            ],
            "neg": [
                r"yalnız", r"sessiz", r"köşem", r"çekingen", r"utangaç", r"içe kapanık",
                r"kalabalıktan kaç", r"sakin", r"kendi başıma", r"konuşmaktan çekin", r"sus",
                r"içime at", r"uzak dur"
            ]
        },
        "agreeableness": {
            "pos": [
                r"şefkat", r"merhamet", r"yardım", r"affet", r"kırmak istemem", r"sevg",
                r"dost", r"paylaş", r"fedakâr", r"iyilik", r"anlayış", r"hoşgör", r"kalp",
                r"üzülmesine", r"koru", r"kıyamam", r"helal", r"teşekkür", r"samimi", r"güven",
                r"tatlı dil", r"gönül al"
            ],
            "neg": [
                r"tahammül edemem", r"kavga", r"sert", r"intikam", r"öfke",
                r"nefret", r"ez", r"acımam", r"hak et", r"yüzüne vur",
                r"küstah", r"bencil", r"çıkarcı", r"acımasız", r"kin", r"aşağıla", r"küçümse"
            ]
        },
        "neuroticism": {
            "pos": [
                r"kork", r"kaygı", r"endişe", r"hüzün", r"keder", r"ağla", r"bunal",
                r"stres", r"üzül", r"yalnızlık", r"çaresiz", r"şüphe", r"titri", r"sinir",
                r"kırılgan", r"dert", r"acı", r"felaket", r"melankoli", r"uyuyam", r"içim kan ağla",
                r"kırgın", r"gözyaşı", r"yaralı"
            ],
            "neg": [
                r"soğukkanlı", r"sakin", r"rahat", r"huzur", r"güçlü", r"sarsılmaz",
                r"korkmam", r"metanet", r"dengeli", r"umut", r"ferah", r"endişelenmem"
            ]
        }
    }

    exclam = text.count("!")
    ellipsis = text.count("...")
    quest = text.count("?")

    user_scores = {}
    for trait, poles in signals.items():
        pos_hits = sum(len(re.findall(p, t_low)) for p in poles["pos"])
        neg_hits = sum(len(re.findall(p, t_low)) for p in poles["neg"])

        bonus = 0.0
        if trait == "extraversion":
            bonus = 0.35 * min(exclam, 3) - 0.25 * min(ellipsis, 3)
        elif trait == "neuroticism":
            bonus = 0.35 * min(ellipsis, 3) + 0.15 * min(quest, 3)
        elif trait == "openness":
            bonus = 0.20 * min(quest, 3)

        net = (pos_hits - 1.15 * neg_hits) + bonus
        score = 0.50 + 0.42 * np.tanh(0.65 * net)
        user_scores[trait] = float(np.clip(score, 0.12, 0.94))

    return user_scores

def find_literary_twins(user_scores, preds):
    """
    Z-Skor Standartlaştırılmış Öklid Mesafesi ile karakter eşleştirmesi.
    Her karakterin kendine özgü baskın özelliklerini dikkate alır.
    """
    char_names = list(preds.keys())
    mat = np.array([[preds[c][t] for t in TRAITS] for c in char_names])

    means = np.mean(mat, axis=0)
    stds = np.std(mat, axis=0) + 1e-6

    u_vec = np.array([user_scores[t] for t in TRAITS])
    u_z = (u_vec - means) / stds

    results = []
    for idx, c_name in enumerate(char_names):
        c_vec = mat[idx]
        c_z = (c_vec - means) / stds

        z_dist = np.linalg.norm(u_z - c_z) / np.sqrt(len(TRAITS))
        raw_mae = np.mean(np.abs(u_vec - c_vec))

        combined_dist = 0.55 * z_dist + 0.45 * (raw_mae * 4.0)
        match_pct = float(np.clip(100.0 * np.exp(-0.75 * combined_dist), 35.0, 99.2))

        # En çok örtüşen 2 özelliği bul
        trait_diffs = [(t, abs(user_scores[t] - preds[c_name][t])) for t in TRAITS]
        trait_diffs.sort(key=lambda x: x[1])
        common_traits = [TRAIT_INFO[trait_diffs[0][0]][0], TRAIT_INFO[trait_diffs[1][0]][0]]

        results.append({
            "char": c_name,
            "similarity": round(match_pct, 1),
            "common_traits": common_traits
        })

    results.sort(key=lambda x: -x["similarity"])
    return results

# --- GRAFİK FONKSİYONLARI ---
def make_clean_radar(name1, scores1, name2=None, scores2=None, is_dark=False):
    labels = [TRAIT_INFO[t][0] for t in TRAITS]
    labels_closed = labels + [labels[0]]

    v1 = [scores1[t] * 100 for t in TRAITS]
    v1_closed = v1 + [v1[0]]

    fig = go.Figure()

    if is_dark:
        c1_line = "#E07A5F"
        c1_fill = "rgba(224, 122, 95, 0.25)"
        c2_line = "#5BA88C"
        c2_fill = "rgba(91, 168, 140, 0.20)"
        bg_polar = "#171614"
        grid_color = "#302C26"
        line_color = "#3E3932"
        tick_color = "#999388"
        label_color = "#EDE8DF"
    else:
        c1_line = "#9E3B24"
        c1_fill = "rgba(158, 59, 36, 0.20)"
        c2_line = "#3B6E5C"
        c2_fill = "rgba(59, 110, 92, 0.14)"
        bg_polar = "#FCFBF9"
        grid_color = "#E6E0D5"
        line_color = "#D8D0C5"
        tick_color = "#78736A"
        label_color = "#1F1E1B"

    if scores2 is not None:
        v2 = [scores2[t] * 100 for t in TRAITS]
        v2_closed = v2 + [v2[0]]
        fig.add_trace(go.Scatterpolar(
            r=v2_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=c2_fill,
            line=dict(color=c2_line, width=2.2, dash="dot"),
            name=name2
        ))

    fig.add_trace(go.Scatterpolar(
        r=v1_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=c1_fill,
        line=dict(color=c1_line, width=2.8),
        marker=dict(size=7, color=c1_line),
        name=name1
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=bg_polar,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticksuffix="%",
                gridcolor=grid_color,
                tickfont=dict(size=10, color=tick_color),
            ),
            angularaxis=dict(
                gridcolor=grid_color,
                linecolor=line_color,
                tickfont=dict(family="Inter", size=12, color=label_color),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=45, t=25, b=35),
        height=370,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter", size=12, color=label_color)
        )
    )
    return fig

def main():
    # --- TEMA YÖNETİMİ (AÇIK / KOYU TEMA) ---
    if "app_theme" not in st.session_state:
        st.session_state["app_theme"] = "light"

    def set_theme_top():
        val = st.session_state.get("theme_top_radio", "")
        st.session_state["app_theme"] = "dark" if "Koyu" in val else "light"
        st.session_state["theme_sidebar_radio"] = "🌙 Koyu Tema (Gece Modu)" if "Koyu" in val else "📖 Açık Tema (Kitap Kağıdı)"

    def set_theme_side():
        val = st.session_state.get("theme_sidebar_radio", "")
        st.session_state["app_theme"] = "dark" if "Koyu" in val else "light"
        st.session_state["theme_top_radio"] = "🌙 Koyu Tema" if "Koyu" in val else "📖 Açık Tema"

    is_dark = (st.session_state["app_theme"] == "dark")
    inject_clean_literary_css(is_dark=is_dark)

    # --- YAN MENÜ (SIDEBAR) ---
    with st.sidebar:
        st.markdown("### 🎨 Görünüm Teması")
        st.radio(
            "Temayı Seçin:",
            options=["📖 Açık Tema (Kitap Kağıdı)", "🌙 Koyu Tema (Gece Modu)"],
            index=1 if is_dark else 0,
            key="theme_sidebar_radio",
            on_change=set_theme_side
        )
        st.markdown("---")
        st.markdown("### 📚 İncelenen Romanlar")
        st.markdown("""
        * **Reşat Nuri Güntekin**
          * *Çalıkuşu* (1922)
          * *Yaprak Dökümü* (1930)
        * **Sabahattin Ali**
          * *İçimizdeki Şeytan* (1940)
          * *Kürk Mantolu Madonna* (1943)
        """)
        st.markdown("---")
        st.markdown("### 🧠 NLP & Psikometri")
        st.markdown("""
        * **Dil Modeli:** `BERTurk`
        * **Kuram:** Beş Faktör Kişilik Kuramı (Big Five)
        * **Kapsam:** 4 Klasik Eser, 20 Karakter
        """)

    gt, preds, emotions, core_dialogues, extracted_contexts = load_data()

    # --- ÜST TEMA ÇUBUĞU ---
    col_brand, col_theme = st.columns([2.6, 1.4], vertical_alignment="center")
    with col_brand:
        render_html(f"""
        <div style="font-size:0.83rem; font-weight:600; letter-spacing:0.06em; color:{'#A8A298' if is_dark else '#78736A'}; padding: 4px 0;">
            🇹🇷 TÜRK EDEBİYATI • PSİKOMETRİK KARAKTER ANALİZİ
        </div>
        """)
    with col_theme:
        st.radio(
            "Görünüm Teması",
            options=["📖 Açık Tema", "🌙 Koyu Tema"],
            index=1 if is_dark else 0,
            key="theme_top_radio",
            on_change=set_theme_top,
            horizontal=True,
            label_visibility="collapsed"
        )

    # --- ÜST BAŞLIK ---
    render_html("""
    <div class="site-header">
        <div class="site-badge">TÜRK EDEBİYATI KARAKTER PORTRELERİ</div>
        <h1 class="site-title">Roman Karakterlerinin Ruh ve Kişilik Haritası</h1>
        <p class="site-desc">
            Reşat Nuri Güntekin ve Sabahattin Ali'nin dört klasik eserindeki 20 unutulmaz karakterin; 
            romandaki konuşmaları, iç sesleri ve davranışları üzerinden oluşturulmuş psikolojik portreleri.
        </p>
    </div>
    """)

    # --- 4 SEKME ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Karakter İncelemesi",
        "⚖️ İki Karakteri Karşılaştır",
        "✨ Ben Hangi Roman Karakteriyim?",
        "📚 Romanlar ve Genel Tablo"
    ])

    # =========================================================================
    # SEKME 1: KARAKTER İNCELEMESİ
    # =========================================================================
    with tab1:
        col_n, col_c = st.columns([1, 1.3])
        with col_n:
            selected_novel = st.selectbox(
                "1. İncelemek İstediğiniz Romanı Seçin:",
                options=list(NOVELS.keys()),
                format_func=lambda k: f"{NOVELS[k]['title']} — {NOVELS[k]['author']} ({NOVELS[k]['year']})"
            )
        novel_chars = [c for c, info in gt.items() if info.get("novel", "calikusu") == selected_novel]
        with col_c:
            selected_char = st.selectbox(
                "2. Karakter Seçin:",
                options=novel_chars,
                format_func=lambda c: f"{c.replace('_', ' ')} ({CHARACTERS_INFO.get(c, ('', '', ''))[1]})"
            )

        if not selected_char or selected_char not in novel_chars:
            selected_char = novel_chars[0]

        char_display = selected_char.replace("_", " ")
        novel_title, char_role, char_bio = CHARACTERS_INFO.get(selected_char, ("", "", ""))
        char_pred = preds[selected_char]
        char_gt = gt[selected_char]
        char_emo = emotions.get(selected_char, {})
        char_quotes = extracted_contexts.get(selected_char, [])

        # Sol: Karakter Kimliği ve Kişilik Grafiği | Sağ: 5 Kişilik Özelliği Açıklamalı Barlar
        left_col, right_col = st.columns([1.05, 1.15], gap="large")

        with left_col:
            render_html(f"""
            <div class="clean-card">
                <div class="card-tag">{novel_title} • {NOVELS[selected_novel]['author']}</div>
                <h2 class="card-title">{char_display}</h2>
                <div class="card-subtitle">{char_role}</div>
                <div class="card-body">{char_bio}</div>
            </div>
            """)

            radar_fig = make_clean_radar(
                f"{char_display} (Metin Analizi)",
                char_pred,
                "Edebi Referans Profili",
                char_gt,
                is_dark=is_dark
            )
            st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

        with right_col:
            bars_parts = [
                '<div class="clean-card">',
                '<div class="card-tag">PSİKOLOJİK PORTRE</div>',
                f'<h3 class="card-title" style="font-size:1.35rem; margin-bottom:1.1rem;">{char_display} Kişilik Özellikleri</h3>'
            ]
            for t in TRAITS:
                t_name, t_desc = TRAIT_INFO[t]
                pct = int(round(char_pred[t] * 100))
                level_label = "Çok Yüksek" if pct >= 75 else ("Yüksek" if pct >= 60 else ("Orta Düzey" if pct >= 42 else "Düşük"))
                bars_parts.append(
                    f'<div class="trait-item">'
                    f'<div class="trait-top">'
                    f'<span class="trait-name">{t_name}</span>'
                    f'<span class="trait-score">%{pct} ({level_label})</span>'
                    f'</div>'
                    f'<div class="trait-bar-bg">'
                    f'<div class="trait-bar-fill" style="width: {pct}%;"></div>'
                    f'</div>'
                    f'<div class="trait-hint">{t_desc}</div>'
                    f'</div>'
                )
            bars_parts.append('</div>')
            render_html("".join(bars_parts))

            # Baskın Duygular Özeti
            active_emotion_info = EMOTION_INFO_DARK if is_dark else EMOTION_INFO
            sorted_emos = sorted(char_emo.items(), key=lambda x: -x[1])[:3]
            emo_pills = "".join([
                f'<span style="display:inline-block; background:{active_emotion_info[k][1]}22; color:{active_emotion_info[k][1]}; border:1px solid {active_emotion_info[k][1]}50; padding:6px 12px; border-radius:20px; font-size:0.85rem; font-weight:600; margin-right:8px; margin-bottom:6px;">{active_emotion_info[k][0]}: %{int(v*100)}</span>'
                for k, v in sorted_emos if k in active_emotion_info
            ])
            render_html(f"""
            <div class="clean-card" style="padding:1.2rem 1.5rem;">
                <div class="card-tag">KONUŞMALARINDAKİ BASKIN DUYGULAR</div>
                <div style="margin-top:0.5rem;">{emo_pills}</div>
            </div>
            """)

        # Romandan Örnek Diyaloglar
        render_html(f"""
        <div class="clean-card" style="margin-top:0.5rem;">
            <div class="card-tag">ROMANDAN DOĞRUDAN ALINTILAR</div>
            <h3 class="card-title" style="font-size:1.35rem; margin-bottom:0.3rem;">{char_display} Karakterinin Romandaki Sözleri ve Sahne Bağlamı</h3>
            <p class="card-subtitle" style="margin-bottom:0;">
                Karakterin kişilik profili çıkarılırken roman metninden otomatik yakalanan örnek konuşmaları:
            </p>
        </div>
        """)

        quotes_to_show = char_quotes[:4] if char_quotes else [
            {"utterance": q, "narrator_context": ""} for q in core_dialogues.get(selected_char, [])[:4]
        ]
        q_col1, q_col2 = st.columns(2, gap="medium")
        for i, q_item in enumerate(quotes_to_show):
            ctx_str = q_item.get("narrator_context", "").strip()
            box_html = (
                f'<div class="dialogue-box">'
                f'<div class="dialogue-quote">“{q_item["utterance"]}”</div>'
                f'{(f"<div class=\'dialogue-context\'><strong>Sahne Bağlamı:</strong> {ctx_str[:180]}...</div>") if ctx_str else ""}'
                f'</div>'
            )
            if i % 2 == 0:
                q_col1.markdown(box_html, unsafe_allow_html=True)
            else:
                q_col2.markdown(box_html, unsafe_allow_html=True)

    # =========================================================================
    # SEKME 2: İKİ KARAKTERİ KARŞILAŞTIR
    # =========================================================================
    with tab2:
        render_html("""
        <div class="clean-card">
            <div class="card-tag">KARŞILAŞTIRMALI EDEBİYAT</div>
            <h2 class="card-title">İki Roman Karakterini Yan Yana Karşılaştırın</h2>
            <p class="card-body">
                Aynı romandan veya farklı romanlardan iki karakter seçerek hangi huylarının benzediğini, 
                hangi yönleriyle birbirine zıt olduğunu tek bakışta inceleyebilirsiniz.
            </p>
        </div>
        """)

        all_chars = list(preds.keys())
        c1, c2 = st.columns(2)
        with c1:
            char_1 = st.selectbox(
                "1. Karakteri Seçin:",
                options=all_chars,
                index=all_chars.index("Feride") if "Feride" in all_chars else 0,
                format_func=lambda c: f"{c.replace('_', ' ')} ({CHARACTERS_INFO[c][0]})"
            )
        with c2:
            char_2 = st.selectbox(
                "2. Karakteri Seçin:",
                options=all_chars,
                index=all_chars.index("Raif_Efendi") if "Raif_Efendi" in all_chars else 1,
                format_func=lambda c: f"{c.replace('_', ' ')} ({CHARACTERS_INFO[c][0]})"
            )

        left_cmp, right_cmp = st.columns([1.1, 1.1], gap="large")
        with left_cmp:
            cmp_fig = make_clean_radar(
                char_1.replace("_", " "),
                preds[char_1],
                char_2.replace("_", " "),
                preds[char_2],
                is_dark=is_dark
            )
            st.plotly_chart(cmp_fig, use_container_width=True, config={"displayModeBar": False})

        with right_cmp:
            if char_1 == char_2:
                cmp_body = (
                    f"Aynı karakteri seçtiniz (<strong>{char_1.replace('_', ' ')}</strong>). "
                    "Karşılaştırma grafiğinde iki farklı karakter arasındaki ayrışmaları ve "
                    "benzerlikleri incelemek için lütfen 2. seçim kutusundan farklı bir karakter belirleyin."
                )
            else:
                diffs = [(t, abs(preds[char_1][t] - preds[char_2][t])) for t in TRAITS]
                diffs.sort(key=lambda x: x[1])
                most_similar_trait = TRAIT_INFO[diffs[0][0]][0]
                most_diff_trait = TRAIT_INFO[diffs[-1][0]][0]
                cmp_body = (
                    f"• <strong>En Çok Benzeştikleri Yön:</strong> İki karakter birbirine en çok <strong>{most_similar_trait}</strong> özelliğinde yakındır.<br><br>"
                    f"• <strong>En Belirgin Farkları:</strong> İki karakteri birbirinden ayıran en keskin çizgi <strong>{most_diff_trait}</strong> boyutundadır.<br><br>"
                    f"• <strong>{char_1.replace('_', ' ')} ({CHARACTERS_INFO[char_1][0]}):</strong> {CHARACTERS_INFO[char_1][2]}<br><br>"
                    f"• <strong>{char_2.replace('_', ' ')} ({CHARACTERS_INFO[char_2][0]}):</strong> {CHARACTERS_INFO[char_2][2]}"
                )

            render_html(f"""
            <div class="clean-card">
                <div class="card-tag">KARŞILAŞTIRMA ÖZETİ</div>
                <h3 class="card-title" style="font-size:1.35rem;">{char_1.replace('_', ' ')} &amp; {char_2.replace('_', ' ')}</h3>
                <p class="card-body">{cmp_body}</p>
            </div>
            """)

    # =========================================================================
    # SEKME 3: BEN HANGİ ROMAN KARAKTERİYİM? (EDEBİ İKİZ)
    # =========================================================================
    with tab3:
        render_html("""
        <div class="clean-card">
            <div class="card-tag">ETKİLEŞİMLİ KİŞİLİK EŞLEŞTİRME</div>
            <h2 class="card-title">Ben Hangi Roman Karakteriyim?</h2>
            <p class="card-body">
                Kendinizi, hayata bakışınızı anlatan birkaç cümle yazın veya aşağıdaki <strong>hazır örneklerden birine</strong> tıklayın. 
                Sistem yazdığınız cümlelerin üslubunu ve duygusunu inceleyerek 4 romandaki 20 karakter arasından 
                size en çok benzeyen karakterleri bulur.
            </p>
        </div>
        """)

        presets = {
            "Kendi Metnimi Yazacağım": "",
            "Örnek 1: İçe Kapanık, Sanatkâr ve Duygusal (Raif Efendi Tipi)": (
                "İnsanların arasına karışmaktan çekiniyorum, sessizce kendi köşemde durup kitap okumak ve "
                "resim yapmak bana yetiyor. Kimseye yük olmak istemem, içimde derin bir yalnızlık ve keder var..."
            ),
            "Örnek 2: Neşeli, Cesur ve Gururlu (Feride Tipi)": (
                "Neşemi asla kaybetmem; içim kan ağlasa da etrafıma gülümserim. Kimseye boyun eğmem, haksızlığa gelemem; "
                "özgürlüğüme düşkünüm ve kendi ayaklarımın üzerinde dimdik dururum!"
            ),
            "Örnek 3: İlkeli, Sorumluluk Sahibi ve Fedakâr (Şevket / Ali Rıza Bey Tipi)": (
                "Bir insanın hayattaki en büyük mirası namusu, doğruluğu ve vazife bilincidir. Ailem ve sevdiklerim "
                "için gece gündüz sabırla çalışır, her türlü fedakârlığı seve seve yaparım."
            ),
            "Örnek 4: Eğlenceye, Lükse Düşkün ve Sabırsız (Ferhunde Tipi)": (
                "Çok sıkıldım bu evin kasvetinden! Ben hiçbir iş yapmaya üşeniyorum, sadece güzel giyinip sosyeteye "
                "karışmak, dans etmek ve gönlümce eğlenmek istiyorum!"
            )
        }

        # Session state başlatma
        default_preset = "Örnek 1: İçe Kapanık, Sanatkâr ve Duygusal (Raif Efendi Tipi)"
        if "user_text" not in st.session_state:
            st.session_state["user_text"] = presets[default_preset]
        if "preset_radio" not in st.session_state:
            st.session_state["preset_radio"] = default_preset

        def on_preset_change():
            sel = st.session_state.get("preset_radio")
            if sel:
                st.session_state["user_text"] = presets[sel]

        st.radio(
            "Hızlı Denemek İçin Bir Örnek Seçin veya Kendi Cümlenizi Yazın:",
            options=list(presets.keys()),
            key="preset_radio",
            on_change=on_preset_change,
            horizontal=True
        )

        user_input = st.text_area(
            "Düşüncelerinizi veya kendinizi anlatan birkaç cümle yazın:",
            key="user_text",
            height=110,
            placeholder="Örn: Kalabalıklardan ziyade kendi köşemde kitaplarımla baş başa kalmayı ve üretmeyi severim..."
        )

        find_clicked = st.button("Bana En Çok Benzeyen Karakteri Bul", type="primary")
        if find_clicked:
            if not user_input.strip():
                st.warning("⚠️ Lütfen kendinizi anlatan en az bir cümle yazın veya yukarıdaki hazır örneklerden birini seçin.")
                st.session_state["has_twin_result"] = False
            else:
                user_scores = analyze_short_user_text(user_input)
                matches = find_literary_twins(user_scores, preds)
                st.session_state["twin_user_scores"] = user_scores
                st.session_state["twin_matches"] = matches
                st.session_state["has_twin_result"] = True

        if st.session_state.get("has_twin_result"):
            user_scores = st.session_state["twin_user_scores"]
            matches = st.session_state["twin_matches"]
            top1 = matches[0]
            top2 = matches[1]
            top3 = matches[2]
            c1_name = top1["char"]
            c1_novel, c1_role, c1_bio = CHARACTERS_INFO[c1_name]

            r_col1, r_col2 = st.columns([1.15, 1.1], gap="large")
            border_card = "#E07A5F" if is_dark else "#C86D51"
            with r_col1:
                render_html(f"""
                <div class="clean-card" style="border: 2px solid {border_card};">
                    <div class="card-tag">EN YÜKSEK UYUM (%{top1['similarity']})</div>
                    <h2 class="card-title">Edebi İkiziniz: {c1_name.replace('_', ' ')}</h2>
                    <div class="card-subtitle">{c1_novel} — {c1_role}</div>
                    <p class="card-body">{c1_bio}</p>
                    <div class="twin-reason-box">
                        <strong>Neden Eşleştiniz?</strong><br>
                        Yazdığınız metindeki <strong>{top1['common_traits'][0]}</strong> ve 
                        <strong>{top1['common_traits'][1]}</strong> düzeyiniz, {c1_name.replace('_', ' ')} karakterinin 
                        romandaki kişiliğiyle büyük ölçüde örtüşüyor.
                    </div>
                </div>
                """)

                render_html(f"""
                <div class="clean-card" style="padding:1.1rem 1.4rem;">
                    <div class="card-tag">SIZE YAKIN DİĞER KARAKTERLER</div>
                    <div class="twin-sub-card">
                        <strong>2. {top2['char'].replace('_', ' ')}</strong> ({CHARACTERS_INFO[top2['char']][0]}) — <strong>%{top2['similarity']} Uyum</strong><br>
                        <span class="twin-sub-hint">Ortak Yön: {top2['common_traits'][0]}, {top2['common_traits'][1]}</span>
                        <hr class="twin-sub-divider">
                        <strong>3. {top3['char'].replace('_', ' ')}</strong> ({CHARACTERS_INFO[top3['char']][0]}) — <strong>%{top3['similarity']} Uyum</strong><br>
                        <span class="twin-sub-hint">Ortak Yön: {top3['common_traits'][0]}, {top3['common_traits'][1]}</span>
                    </div>
                </div>
                """)

            with r_col2:
                twin_fig = make_clean_radar(
                    "Sizin Metniniz",
                    user_scores,
                    f"{c1_name.replace('_', ' ')} ({c1_novel})",
                    preds[c1_name],
                    is_dark=is_dark
                )
                st.plotly_chart(twin_fig, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # SEKME 4: ROMANLAR VE GENEL TABLO
    # =========================================================================
    with tab4:
        render_html("""
        <div class="clean-card">
            <div class="card-tag">KÜLLİYAT ÖZETİ</div>
            <h2 class="card-title">İncelenen 4 Roman ve 20 Karakterin Toplu Tablosu</h2>
            <p class="card-body">
                Çalışmada yer alan dört romanın genel özeti ve 20 karakterin beş temel kişilik özelliğindeki yüzdelik puanları:
            </p>
        </div>
        """)

        n_cols = st.columns(4)
        for idx, (n_key, n_info) in enumerate(NOVELS.items()):
            with n_cols[idx]:
                render_html(f"""
                <div class="clean-card" style="min-height: 210px;">
                    <div class="card-tag">{n_info['year']}</div>
                    <h3 class="card-title" style="font-size:1.25rem;">{n_info['title']}</h3>
                    <div class="card-subtitle">{n_info['author']}</div>
                    <div class="card-body" style="font-size:0.86rem;">{n_info['summary']}</div>
                </div>
                """)

        table_rows = []
        for c_name, p_vals in preds.items():
            c_nov, c_role, _ = CHARACTERS_INFO.get(c_name, ("", "", ""))
            table_rows.append({
                "Karakter": c_name.replace("_", " "),
                "Roman": c_nov,
                "Rolü": c_role,
                "Yeniliğe Açıklık": f"%{int(p_vals['openness']*100)}",
                "Sorumluluk": f"%{int(p_vals['conscientiousness']*100)}",
                "Dışadönüklük": f"%{int(p_vals['extraversion']*100)}",
                "Uyumluluk / Şefkat": f"%{int(p_vals['agreeableness']*100)}",
                "Duygusal Hassasiyet": f"%{int(p_vals['neuroticism']*100)}"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Jüri / Teknik İnceleme
        with st.expander("🔬 Bilimsel Yöntem ve Model Doğruluk Verileri (Jüri / Teknik İnceleme İçin)"):
            y_t, y_p = [], []
            for c in gt:
                if c in preds:
                    y_t.extend([gt[c][t] for t in TRAITS])
                    y_p.extend([preds[c][t] for t in TRAITS])
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            cos_sim = (1.0 - cosine(y_t, y_p)) * 100
            sp_r, _ = spearmanr(y_t, y_p)

            st.markdown(f"""
            * **Kullanılan Dil Modeli:** `dbmdz/bert-base-turkish-cased` (BERTurk) + Bağlam Duyarlı Psikodilbilimsel Çıkarım
            * **Doğrulama Yöntemi:** *Leave-One-Novel-Out Cross-Validation* (Her roman, diğer 3 roman üzerinde eğitilen modelle sıfır veri sızıntısıyla test edilmiştir).
            * **Ortalama Mutlak Hata (MAE):** `{mae:.4f}` *(Hedef: < 0.15)*
            * **Hata Karekökü (RMSE):** `{rmse:.4f}`
            * **Kosinüs Profil Benzerliği:** `%{cos_sim:.2f}` *(Hedef: >= %85)*
            * **Spearman Sıralama Korelasyonu:** `+{sp_r:.4f}`
            """)

if __name__ == "__main__":
    main()