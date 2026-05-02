import re

# --- CONFIG ---
ACRONYM_MAP_NE = {
    'A': 'ए', 'B': 'बी', 'C': 'सी', 'D': 'डी', 'E': 'ई', 'F': 'एफ', 'G': 'जी',
    'H': 'यच', 'I': 'आइ', 'J': 'जे', 'K': 'के', 'L': 'एल', 'M': 'एम', 'N': 'एन',
    'O': 'ओ', 'P': 'पी', 'Q': 'क्यू', 'R': 'आर', 'S': 'एस', 'T': 'टी', 'U': 'यू',
    'V': 'भी', 'W': 'डब्लू', 'X': 'एक्स', 'Y': 'वाई', 'Z': 'जेड'
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
    70: 'सत्तरी', 71: 'एकहत्तर', 72: 'बहत्तर', 73: 'त्रिहत्तर', 74: 'चौहत्तर', 75: 'पचहत्तर', 76: 'छयत्तर', 77: 'सतहत्तर', 78: 'अठहत्तर', 79: 'उनन्साठी',
    80: 'असी', 81: 'एकासी', 82: 'बयासी', 83: 'त्रियासी', 84: 'चौरासी', 85: 'पचासी', 86: 'छयासी', 87: 'सतासी', 88: 'अठासी', 89: 'उनानब्बे',
    90: 'नब्बे', 91: 'एकानब्बे', 92: 'बयानब्बे', 93: 'त्रियानब्बे', 94: 'चौरानब्बे', 95: 'पन्चानब्बे', 96: 'छयानब्बे', 97: 'सन्तानब्बे', 98: 'अन्ठानब्बे', 99: 'उनन्सय'
}

ENGLISH_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ENGLISH_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

NEPALI_DIGITS = "०१२३४५६७८९"
ENGLISH_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(NEPALI_DIGITS, ENGLISH_DIGITS)

def number_to_nepali(n):
    if n < 100: return NEPALI_NUMS.get(n, str(n))
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
    if n == 0: return "zero"
    if n < 10: return ENGLISH_ONES[n]
    if n < 20: return ENGLISH_TEENS[n-10]
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
        whole_text = number_to_nepali(int(whole)) if lang == "ne" else number_to_english(int(whole))
        point_word = "दशमलव" if lang == "ne" else "point"
        frac_words = [NEPALI_NUMS[int(d)] if lang == "ne" else ENGLISH_ONES[int(d)] for d in frac]
        return f"{whole_text} {point_word} {' '.join(frac_words)}"
    try:
        val = int(num_str)
        return number_to_nepali(val) if lang == "ne" else number_to_english(val)
    except:
        return num_str

def sanitize_text(text, lang="ne"):
    # 1. Immediate whitespace cleanup
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Normalize curly quotes/apostrophes early
    text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
    
    # 2. Markdown removal
    text = re.sub(r'[*_]{1,3}', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^[#>\-\+\*]\s+', '', text, flags=re.MULTILINE)

    # 3. Units mapping (Number + Unit)
    units = {
        'm': ('मिटर', 'meters'), 'km': ('किलोमिटर', 'kilometers'),
        'kg': ('किलोग्राम', 'kilograms'), 'cm': ('सेन्टिमिटर', 'centimeters'),
        'ft': ('फिट', 'feet'), 'in': ('इन्च', 'inches'),
        'g': ('ग्राम', 'grams'), 'l': ('लिटर', 'liters'),
    }
    for unit, (ne_word, en_word) in units.items():
        pattern = rf'([0-9०-९,.]+)\s?{unit}\b'
        word = ne_word if lang == "ne" else en_word
        text = re.sub(pattern, lambda m: f"{sanitize_numbers(m.group(1), lang)} {word}", text)

    # 4. Symbols
    if lang == "ne":
        text = re.sub(r'([0-9०-९,.]+)\s?%', lambda m: f"{sanitize_numbers(m.group(1), lang)} प्रतिशत", text)
        symbol_map = {'&': 'र', '@': 'एट', '#': 'ह्यास', '$': 'डलर', '/': 'स्ल्याश'}
    else:
        text = re.sub(r'([0-9०-९,.]+)\s?%', lambda m: f"{sanitize_numbers(m.group(1), lang)} percent", text)
        symbol_map = {'&': 'and', '@': 'at', '#': 'hash', '$': 'dollars', '/': 'slash'}
        
    for sym, word in symbol_map.items():
        text = text.replace(sym, f" {word} ")

    # 5. Time (10:30 AM)
    time_regex = r'\b([0-9०-९]{1,2}):([0-9०-९]{2})\s?(AM|PM|am|pm)?\b'
    def replace_time(match):
        h, m, suffix = match.group(1), match.group(2), (match.group(3) or "").upper()
        h_word, m_word = sanitize_numbers(h, lang), sanitize_numbers(m, lang)
        if suffix:
            suffix_word = suffix if lang == "en" else " ".join([ACRONYM_MAP_NE.get(c, c) for c in suffix])
            return f"{h_word} {m_word} {suffix_word}"
        return f"{h_word} {m_word}"
    text = re.sub(time_regex, replace_time, text)

    # 6. Phone Numbers (Pairs - only for Nepali)
    if lang == "ne":
        phone_regex = r'\b((?:98|97|96)[0-9०-९\s-]{8,11}|0[1-9][0-9०-९\s-]{6,10})\b'
        def replace_phone(match):
            raw = match.group(0)
            digits = raw.translate(DIGIT_MAP).replace("-", "").replace(" ", "")
            if not (7 <= len(digits) <= 11): return raw
            res, i = [], 0
            while i < len(digits):
                if i + 1 < len(digits):
                    pair_str = digits[i:i+2]
                    if pair_str[0] == "0":
                        res.append(f"शून्य {NEPALI_NUMS[int(pair_str[1])]}")
                    else:
                        res.append(number_to_nepali(int(pair_str)))
                    i += 2
                else:
                    res.append(NEPALI_NUMS[int(digits[i])]); i += 1
            return " " + " ".join(res) + " "
        text = re.sub(phone_regex, replace_phone, text)

    # 7. Unified Currencies
    currency_regex = r'(Rs\.?|रू\.?|रु\.?)?\s?([0-9०-९,]+(?:\.[0-9०-९,]+)?)\s?(रुपैयाँ|rupees)?'
    def replace_currency(match):
        if not match.group(1) and not match.group(3): return match.group(0)
        word = "रुपैयाँ" if lang == "ne" else "rupees"
        return f" {sanitize_numbers(match.group(2), lang)} {word} "
    text = re.sub(currency_regex, replace_currency, text)

    # 8. Remaining Numbers
    text = re.sub(r'\b[0-9०-९,]+(?:\.[0-9०-९,]+)?\b', lambda m: sanitize_numbers(m.group(0), lang), text)

    # 9. Acronyms
    if lang == "ne":
        # For Nepali, expand capital letters to Devanagari phonetics
        text = re.sub(r'\b([a-z])\b', lambda m: m.group(1).upper(), text)
        def replace_acronym_ne(match):
            acro, suff = match.group(1), match.group(2) or ""
            mapped = " ".join([ACRONYM_MAP_NE.get(c, c) for c in acro])
            return f"{mapped}{suff}"
        text = re.sub(r'\b([A-Z]{1,})([अ-ञा-्]*)\b', replace_acronym_ne, text)
    else:
        # For English, preserve most acronyms as-is, maybe expand some
        # But specifically don't map "S" to "एस"
        pass

    # 10. FINAL SAFETY CLEANUP
    if lang == "ne":
        # Aggressive cleanup for Nepali to prevent halluncinations from punctuation
        text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s।\.?!]', ' ', text)
    else:
        # Lighter cleanup for English - preserve apostrophes, hyphens, and commas
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\'\"]', ' ', text)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    example = "नेपाल दक्षिण एसियामा अवस्थित देश हो।\nयसको जनसंख्या ३ करोड छ। 9841-456132 नम्बरमा सम्पर्क गर्नुहोस्।"
    print(sanitize_text(example, lang="ne"))
