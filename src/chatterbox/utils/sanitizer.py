import re

# --- NEPALI CONFIG ---
ACRONYM_MAP_NE = {
    'A': 'ए', 'B': 'बी', 'C': 'सी', 'D': 'डी', 'E': 'ई', 'F': 'एफ', 'G': 'जी',
    'H': 'यच', 'I': 'आइ', 'J': 'जे', 'K': 'के', 'L': 'एल', 'M': 'एम', 'N': 'एन',
    'O': 'ओ', 'P': 'पी', 'Q': 'क्यू', 'R': 'आर', 'S': 'एस', 'T': 'टी', 'U': 'यू',
    'V': 'भी', 'W': 'डब्लू', 'X': 'एक्स', 'Y': 'वाई', 'Z': 'जेड'
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

ACRONYM_MAP_EN = {
    'A': 'ay', 'B': 'bee', 'C': 'see', 'D': 'dee', 'E': 'ee', 'F': 'ef', 'G': 'gee',
    'H': 'aitch', 'I': 'eye', 'J': 'jay', 'K': 'kay', 'L': 'el', 'M': 'em', 'N': 'en',
    'O': 'oh', 'P': 'pee', 'Q': 'cue', 'R': 'ar', 'S': 'ess', 'T': 'tee', 'U': 'you',
    'V': 'vee', 'W': 'double-u', 'X': 'ex', 'Y': 'why', 'Z': 'zee'
}

# --- ENGLISH CONFIG ---
ENGLISH_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ENGLISH_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

ENGLISH_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
    "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth",
    "11th": "eleventh", "12th": "twelfth", "13th": "thirteenth", "20th": "twentieth",
}

NEPALI_ORDINALS = {
    "१औं": "पहिलो", "२औं": "दोस्रो", "३औं": "तेस्रो", "४औं": "चौथो", "५औं": "पाँचौं",
    "६औं": "छैटौं", "७औं": "सातौं", "८औं": "आठौं", "९औं": "नवौं", "१०औं": "दशौं",
    "१st": "पहिलो", "२nd": "दोस्रो", "३rd": "तेस्रो", "४th": "चौथो",
}

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
    if n < 1000000000000:
        billions, rem = divmod(n, 1000000000)
        res = number_to_english(billions) + " billion"
        return res + " " + number_to_english(rem) if rem > 0 else res
    if n < 1000000000000000:
        trillions, rem = divmod(n, 1000000000000)
        res = number_to_english(trillions) + " trillion"
        return res + " " + number_to_english(rem) if rem > 0 else res
    return str(n)

NEPALI_DIGITS = "०१२३४५६७८९"
ENGLISH_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(NEPALI_DIGITS, ENGLISH_DIGITS)

def sanitize_numbers(num_str, lang="ne"):
    """Converts a number string (decimal or whole) to words."""
    num_str = num_str.translate(DIGIT_MAP).replace(",", "")
    
    if "." in num_str:
        parts = num_str.split(".", 1)
        whole = parts[0] if parts[0] else "0"
        frac = parts[1]
        
        whole_val = int(whole)
        whole_text = number_to_nepali(whole_val) if lang == "ne" else number_to_english(whole_val)
        
        point_word = "दशमलव" if lang == "ne" else "point"
        
        frac_words = []
        for d in frac:
            d_val = int(d)
            word = NEPALI_NUMS[d_val] if lang == "ne" else ENGLISH_ONES[d_val]
            if not word: word = "zero" # handle 0 in fraction for english
            frac_words.append(word)
            
        return f"{whole_text} {point_word} {' '.join(frac_words)}"
    else:
        try:
            val = int(num_str)
            return number_to_nepali(val) if lang == "ne" else number_to_english(val)
        except:
            return num_str

def sanitize_text(text, lang="ne"):
    # 0. Ordinals (1st, 2nd, १औं, २औं)
    if lang == "ne":
        ordinal_ne_regex = r'([०-९0-9,]+)(औं|औ|st|nd|rd|th)'
        def replace_ordinal_ne(match):
            full = match.group(0)
            if full in NEPALI_ORDINALS:
                return NEPALI_ORDINALS[full]
            # Handle mixed digits and commas
            norm_full = full.translate(DIGIT_MAP).replace(",", "")
            # Try mapping with normalized digits (e.g. 1st)
            if norm_full in ENGLISH_ORDINALS:
                pass
            num_str = match.group(1)
            sanitized_num = sanitize_numbers(num_str, lang="ne")
            return f"{sanitized_num}औं"
        text = re.sub(ordinal_ne_regex, replace_ordinal_ne, text)
    else:
        ordinal_en_regex = r'\b([0-9,]+)(st|nd|rd|th)\b'
        def replace_ordinal_en(match):
            full = match.group(0).lower().replace(",", "")
            if full in ENGLISH_ORDINALS:
                return ENGLISH_ORDINALS[full]
            
            num_str = match.group(1).replace(",", "")
            val = int(num_str)
            last_digit = val % 10
            last_two = val % 100
            
            # Special handling for 21st, 22nd, 33rd, etc.
            if last_digit == 1 and last_two != 11:
                base = sanitize_numbers(str(val - 1), lang="en")
                return f"{base}-first" if base != "zero" else "first"
            if last_digit == 2 and last_two != 12:
                base = sanitize_numbers(str(val - 2), lang="en")
                return f"{base}-second" if base != "zero" else "second"
            if last_digit == 3 and last_two != 13:
                base = sanitize_numbers(str(val - 3), lang="en")
                return f"{base}-third" if base != "zero" else "third"

            sanitized_num = sanitize_numbers(num_str, lang="en")
            if sanitized_num.endswith("y"):
                return sanitized_num[:-1] + "ieth"
            return f"{sanitized_num}th"
        text = re.sub(ordinal_en_regex, replace_ordinal_en, text)

    # 1. Unified Currencies (Rs. 100, रु १००, १०० रुपैयाँ, रु १०० रुपैयाँ)
    # This regex looks for (Optional Prefix) (Number) (Optional Suffix)
    # We only treat it as currency if at least one symbol/word is present.
    currency_regex = r'(Rs\.?|रू\.?|रु\.?)?\s?([0-9०-९,]+(?:\.[0-9०-९,]+)?)\s?(रुपैयाँ|रुपिया|रुपियाँ|rupees|rupee)?'
    
    def replace_currency(match):
        prefix = match.group(1)
        num_str = match.group(2)
        suffix = match.group(3)
        
        # If neither prefix nor suffix exists, it's just a normal number, let it be handled by step 2
        if not prefix and not suffix:
            return match.group(0)
            
        sanitized_num = sanitize_numbers(num_str, lang)
        # Use the existing suffix if present, otherwise default to "रुपैयाँ"/"rupees"
        word = suffix if suffix else ("रुपैयाँ" if lang == "ne" else "rupees")
        return f"{sanitized_num} {word}"
    
    text = re.sub(currency_regex, replace_currency, text)

    # 2. Decimal and Whole Numbers (that were not caught as currencies)
    number_regex = r'\b[0-9०-९,]+(?:\.[0-9०-९,]+)?\b'
    def replace_general_number(match):
        return sanitize_numbers(match.group(0), lang)
        
    text = re.sub(number_regex, replace_general_number, text)

    # 3. Acronyms (2+ Caps letters)
    def replace_acronym(match):
        acronym = match.group(0)
        if lang == "ne":
            return " ".join([ACRONYM_MAP_NE.get(char, char) for char in acronym])
        else:
            return " ".join([ACRONYM_MAP_EN.get(char, char) for char in acronym])
    
    text = re.sub(r'\b[A-Z]{2,}\b', replace_acronym, text)
    
    # Final cleanup of extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    test_cases = [
        ("Rs. 1.8", "en"),
        ("रु १.८", "ne"),
        ("150.50 रुपैयाँ", "ne"),
        ("The value is 1.8 and costs Rs. 500", "en"),
        ("यसको मूल्य १.८ छ र रु ५०० पर्छ।", "ne"),
    ]
    for t, l in test_cases:
        print(f"[{l}] '{t}' -> '{sanitize_text(t, lang=l)}'")
