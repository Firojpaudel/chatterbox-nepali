import re

# --- CONFIG ---
ACRONYM_MAP_NE = {
    'A': 'ए', 'B': 'बि', 'C': 'सि', 'D': 'डि', 'E': 'इ', 'F': 'एफ', 'G': 'जि',
    'H': 'यच', 'I': 'आइ', 'J': 'जे', 'K': 'के', 'L': 'एल', 'M': 'एम', 'N': 'एन',
    'O': 'ओ', 'P': 'पि', 'Q': 'क्यु', 'R': 'आर', 'S': 'यस', 'T': 'टि', 'U': 'यु',
    'V': 'भि', 'W': 'डब्लु', 'X': 'यक्स', 'Y': 'वाइ', 'Z': 'जेड'
}

ACRONYM_MAP_EN = {
    'A': 'ay', 'B': 'bee', 'C': 'see', 'D': 'dee', 'E': 'ee', 'F': 'ef', 'G': 'gee',
    'H': 'aitch', 'I': 'eye', 'J': 'jay', 'K': 'kay', 'L': 'el', 'M': 'em', 'N': 'en',
    'O': 'oh', 'P': 'pee', 'Q': 'cue', 'R': 'ar', 'S': 'ess', 'T': 'tee', 'U': 'you',
    'V': 'vee', 'W': 'double-u', 'X': 'ex', 'Y': 'why', 'Z': 'zee'
}

NEPALI_NUMS = {
    0: 'शून्य', 1: 'एक', 2: 'दुई', 3: 'तीन', 4: 'चार', 5: 'पाँच', 6: 'छ', 7: 'सात', 8: 'आठ', 9: 'नौ',
    10: 'दश', 11: 'एघार', 12: 'बाह्र', 13: 'तेह्र', 14: 'चौध', 15: 'पन्ध्र', 16: 'सोह्र', 17: 'सत्र', 18: 'अठार', 19: 'उन्नाइस',
    20: 'बीस', 21: 'एकाइस', 22: 'बाइस', 23: 'तेइस', 24: 'चौबिस', 25: 'पच्चिस', 26: 'छब्बिस', 27: 'सत्ताइस', 28: 'अठ्ठाइस', 29: 'उनन्तीस',
    30: 'तीस', 31: 'एकतीस', 32: 'बत्तीस', 33: 'तेतीस', 34: 'चौंतीस', 35: 'पैंतीस', 36: 'छत्तीस', 37: 'सैंतीस', 38: 'अठतीस', 39: 'उनन्चालीस',
    40: 'चालीस', 41: 'एकचालीस', 42: 'बयालीस', 43: 'त्रिचालीस', 44: 'चवालीस', 45: 'पैंतालीस', 46: 'छयालीस', 47: 'सत्तालीस', 48: 'अठचालीस', 49: 'उनन्पचास',
    50: 'पचास', 51: 'एकाउन्न', 52: 'बाउन्न', 53: 'त्रिपन्न', 54: 'चउन्न', 55: 'पचपन्न', 56: 'छपन्न', 57: 'सन्ताउन्न', 58: 'अन्ठाउन्न', 59: 'उनन्साठी',
    60: 'साठी', 61: 'एकसाठी', 62: 'बासाठी', 63: 'त्रिसाठी', 64: 'चौंठी', 65: 'पैंठी', 66: 'छासाठी', 67: 'सरसाठी', 68: 'अठसाठी', 69: 'उनन्सत्तरी',
    70: 'सत्तरी', 71: 'एकहत्तर', 72: 'बहत्तर', 73: 'त्रिहत्तर', 74: 'चौहत्तर', 75: 'पचहत्तर', 76: 'छयत्तर', 77: 'सतहत्तर', 78: 'अठहत्तर', 79: 'उनासी',
    80: 'असी', 81: 'एकासी', 82: 'बयासी', 83: 'त्रियासी', 84: 'चौरासी', 85: 'पचासी', 86: 'छयासी', 87: 'सतासी', 88: 'अठासी', 89: 'उनानब्बे',
    90: 'नब्बे', 91: 'एकानब्बे', 92: 'बयानब्बे', 93: 'त्रियानब्बे', 94: 'चौरानब्बे', 95: 'पन्चानब्बे', 96: 'छयानब्बे', 97: 'सन्तानब्बे', 98: 'अन्ठानब्बे', 99: 'उनान्सय'
}

ENGLISH_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ENGLISH_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

NEPALI_DIGITS = "०१२३४५६७८९"
ENGLISH_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(NEPALI_DIGITS, ENGLISH_DIGITS)


def number_to_nepali(n):
    if n < 100:
        return NEPALI_NUMS.get(n, str(n))
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        res = NEPALI_NUMS[hundreds] + ' सय'
        return res + ' ' + number_to_nepali(rem) if rem > 0 else res
    if n < 100000:
        thousands, rem = divmod(n, 1000)
        res = number_to_nepali(thousands) + ' हजार'
        return res + ' ' + number_to_nepali(rem) if rem > 0 else res
    if n < 10000000:
        lakhs, rem = divmod(n, 100000)
        res = number_to_nepali(lakhs) + ' लाख'
        return res + ' ' + number_to_nepali(rem) if rem > 0 else res
    if n < 1000000000:
        crores, rem = divmod(n, 10000000)
        res = number_to_nepali(crores) + ' करोड'
        return res + ' ' + number_to_nepali(rem) if rem > 0 else res
    if n < 100000000000:
        arabs, rem = divmod(n, 1000000000)
        res = number_to_nepali(arabs) + ' अर्ब'
        return res + ' ' + number_to_nepali(rem) if rem > 0 else res
    if n < 10000000000000:
        kharabs, rem = divmod(n, 100000000000)
        res = number_to_nepali(kharabs) + ' खर्ब'
        return res + ' ' + number_to_nepali(rem) if rem > 0 else res
    return str(n)


def number_to_english(n):
    if n == 0:
        return "zero"
    if n < 10:
        return ENGLISH_ONES[n]
    if n < 20:
        return ENGLISH_TEENS[n - 10]
    if n < 100:
        tens, rem = divmod(n, 10)
        res = ENGLISH_TENS[tens]
        return res + "-" + ENGLISH_ONES[rem] if rem > 0 else res
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        res = ENGLISH_ONES[hundreds] + " hundred"
        return res + " " + number_to_english(rem) if rem > 0 else res
    if n < 1000000:
        thousands, rem = divmod(n, 1000)
        res = number_to_english(thousands) + " thousand"
        return res + " " + number_to_english(rem) if rem > 0 else res
    if n < 1000000000:
        millions, rem = divmod(n, 1000000)
        res = number_to_english(millions) + " million"
        return res + " " + number_to_english(rem) if rem > 0 else res
    return str(n)


def sanitize_numbers(num_str, lang="ne"):
    num_str = num_str.translate(DIGIT_MAP).replace(",", "")
    if "." in num_str:
        parts = num_str.split(".", 1)
        whole = parts[0] if parts[0] else "0"
        frac = parts[1]
        whole_text = number_to_english(int(whole)) if lang == "en" else number_to_nepali(int(whole))
        point_word = "point" if lang == "en" else "दशमलव"
        frac_words = [ENGLISH_ONES[int(d)] if lang == "en" else NEPALI_NUMS[int(d)] for d in frac]
        return f"{whole_text} {point_word} {' '.join(frac_words)}"
    try:
        val = int(num_str)
        return number_to_english(val) if lang == "en" else number_to_nepali(val)
    except Exception:
        return num_str


def _spell_phone_nepali(digits):
    """Spell a phone number digit-by-digit in Nepali pairs."""
    res = []
    i = 0
    while i < len(digits):
        if i + 1 < len(digits):
            pair = int(digits[i:i + 2])
            if digits[i] == "0":
                res.append(f"शून्य {NEPALI_NUMS[int(digits[i + 1])]}")
            else:
                res.append(number_to_nepali(pair))
            i += 2
        else:
            res.append(NEPALI_NUMS[int(digits[i])])
            i += 1
    return " ".join(res)


def _spell_phone_english(digits):
    """Spell a phone number digit-by-digit in English."""
    return " ".join([ENGLISH_ONES[int(d)] for d in digits])


def sanitize_text(text, lang="ne"):
    # 1. Strip newlines, carriage returns, tabs completely
    # Handle both real newlines and literal \n strings to prevent "N" ghosting
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # 2. Strip all quotation marks
    for ch in ['"', "'", '\u201c', '\u201d', '\u2018', '\u2019', '\u00ab', '\u00bb', '`']:
        text = text.replace(ch, '')

    # 3. Markdown removal
    text = re.sub(r'[*_]{1,3}', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^[#>\-+*]\s+', '', text, flags=re.MULTILINE)
    text = text.replace('```', '')
    # Strip brackets
    text = text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')

    # 4. Units mapping (Number + Unit) — must come before general number sanitization
    units = {
        'km': ('किलोमिटर', 'kilometers'),
        'cm': ('सेन्टिमिटर', 'centimeters'),
        'kg': ('किलोग्राम', 'kilograms'),
        'ft': ('फिट', 'feet'),
        'mm': ('मिलिमिटर', 'millimeters'),
        'm': ('मिटर', 'meters'),
        'g': ('ग्राम', 'grams'),
        'l': ('लिटर', 'liters'),
    }
    for unit, (ne_word, en_word) in units.items():
        word = en_word if lang == "en" else ne_word
        text = re.sub(
            rf'([0-9\u0966-\u096F,.]+)\s?{unit}\b',
            lambda m, w=word: f"{sanitize_numbers(m.group(1), lang)} {w}",
            text
        )

    # 5. Symbols & percentages
    if lang == "en":
        # English ordinals: 1st, 2nd, 3rd, 4th...
        ordinal_map = {"st": "first", "nd": "second", "rd": "third", "th": "th"}
        text = re.sub(
            r'\b(\d+)(st|nd|rd|th)\b',
            lambda m: f"{sanitize_numbers(m.group(1), lang)} {ordinal_map.get(m.group(2), 'th')}",
            text
        )
        text = re.sub(
            r'([0-9\u0966-\u096F,.]+)\s?%',
            lambda m: f"{sanitize_numbers(m.group(1), lang)} percent",
            text
        )
        symbol_map = {'&': 'and', '@': 'at', '#': 'hash', '$': 'dollars', '/': 'slash', '+': 'plus', '=': 'equals'}
    else:
        text = re.sub(
            r'([0-9\u0966-\u096F,.]+)\s?%',
            lambda m: f"{sanitize_numbers(m.group(1), lang)} प्रतिशत",
            text
        )
        symbol_map = {'&': 'र', '@': 'एट', '#': 'ह्यास', '$': 'डलर', '/': 'स्ल्याश', '+': 'प्लस', '=': 'बराबर'}

    for sym, word in symbol_map.items():
        text = text.replace(sym, f" {word} ")

    # 6. Time (10:30 AM)
    def replace_time(match):
        h = sanitize_numbers(match.group(1), lang)
        m = sanitize_numbers(match.group(2), lang)
        suffix = (match.group(3) or "").strip().upper()
        if suffix:
            if lang == "en":
                return f"{h} {m} {suffix}"
            else:
                suffix_ne = " ".join([ACRONYM_MAP_NE.get(c, c) for c in suffix])
                return f"{h} {m} {suffix_ne}"
        return f"{h} {m}"
    text = re.sub(r'\b(\d{1,2}):(\d{2})\s?(AM|PM|am|pm)?\b', replace_time, text)

    # 7. Phone Numbers
    if lang == "en":
        # US/International format
        def replace_phone_en(match):
            digits = re.sub(r'[^0-9]', '', match.group(0))
            return " " + _spell_phone_english(digits) + " "
        text = re.sub(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', replace_phone_en, text)
    else:
        # Nepali phone numbers: 98xx, 97xx, 96xx (mobile) or 01-xxxx (landline)
        def replace_phone_ne(match):
            raw = match.group(0)
            digits = raw.translate(DIGIT_MAP).replace("-", "").replace(" ", "")
            if len(digits) < 7 or len(digits) > 11:
                return raw
            return " " + _spell_phone_nepali(digits) + " "
        text = re.sub(r'\b(?:(?:98|97|96)\d[\d\s-]{6,9}|0[1-9][\d\s-]{5,9})\b', replace_phone_ne, text)

    # 8. Currencies (two specific patterns — NO all-optional groups to prevent backtracking)
    def _currency_replace(num_str):
        word = "rupees" if lang == "en" else "रुपैयाँ"
        return f" {sanitize_numbers(num_str, lang)} {word} "
    # Prefix: "Rs 500", "रु 500", "रू. 500"
    text = re.sub(r'(?:Rs\.?|रू\.?|रु\.?)\s?(\d[\d,]*(?:\.\d+)?)', lambda m: _currency_replace(m.group(1)), text)
    # Suffix: "500 रुपैयाँ", "500 rupees"
    text = re.sub(r'(\d[\d,]*(?:\.\d+)?)\s?(?:रुपैयाँ|rupees)', lambda m: _currency_replace(m.group(1)), text)

    # 9. Hyphens in alphanumeric terms (COVID-19 -> COVID 19)
    text = re.sub(r'([A-Za-z])-(?=[A-Za-z0-9])', r'\1 ', text)
    text = re.sub(r'(\d)-(?=[A-Za-z])', r'\1 ', text)

    # 10. Remaining bare numbers
    text = re.sub(r'\b[\d\u0966-\u096F,]+(?:\.[\d\u0966-\u096F]+)?\b', lambda m: sanitize_numbers(m.group(0), lang), text)

    # 11. Acronyms (regex-based, A-Z map)
    if lang != "en":
        # Promote lone lowercase letters to uppercase first
        text = re.sub(r'\b([a-z])\b', lambda m: m.group(1).upper(), text)
        # Expand consecutive uppercase letters using Devanagari phonetics
        def _expand_acronym_ne(match):
            acro = match.group(1)
            suffix = match.group(2) or ""
            expanded = " ".join([ACRONYM_MAP_NE.get(c, c) for c in acro])
            return f"{expanded}{suffix}"
        text = re.sub(r'\b([A-Z]{2,})(\u0905[\u0900-\u097F]*)?\b', _expand_acronym_ne, text)
    else:
        def _expand_acronym_en(match):
            acro = match.group(1)
            if len(acro) == 1 and acro in ('A', 'I'):
                return acro
            expanded = " ".join([ACRONYM_MAP_EN.get(c, c) for c in acro])
            return f" {expanded} "
        text = re.sub(r'\b([A-Z]{2,})\b', _expand_acronym_en, text)

    # 12. Final safety cleanup
    if lang == "en":
        text = re.sub(r"[^a-zA-Z0-9\s.,?!\-']", ' ', text)
    else:
        # Keep Devanagari + basic Latin + punctuation
        text = re.sub(r'[^a-zA-Z0-9\u0900-\u0D7F\s।.?!,\-]', ' ', text)

    # 13. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # DEBUG: Show final text being synthesized
    print(f"DEBUG [Sanitizer]: '{text}'")

    return text


if __name__ == "__main__":
    # Quick smoke test
    tests = [
        ("9841-456132 नम्बरमा सम्पर्क गर्नुहोस्।", "ne"),
        ("WHO ले COVID-19 लाई महामारी घोषणा गर्यो।", "ne"),
        ("नेपाल हो।\nयसको जनसंख्या ३ करोड छ।", "ne"),
        ("रु 50,00,00,000 लगानी गर्ने योजना।", "ne"),
        ("Call +1-555-123-4567 for info.", "en"),
        ("The GDP grew by 6.5% in 2024.", "en"),
    ]
    for text, lang in tests:
        print(f"[{lang}] {repr(text)}")
        print(f"  => {sanitize_text(text, lang=lang)}")
        print()
