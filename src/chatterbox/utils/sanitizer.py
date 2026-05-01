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
    70: 'सत्तरी', 71: 'एकहत्तर', 72: 'बहत्तर', 73: 'त्रिहत्तर', 74: 'चौहत्तर', 75: 'पचहत्तर', 76: 'छयहत्तर', 77: 'सतहत्तर', 78: 'अठहत्तर', 79: 'उनन्साठी',
    80: 'असी', 81: 'एकासी', 82: 'बयासी', 83: 'त्रियासी', 84: 'चौरासी', 85: 'पचासी', 86: 'छयासी', 87: 'सतासी', 88: 'अठासी', 89: 'उनानब्बे',
    90: 'नब्बे', 91: 'एकानब्बे', 92: 'बयानब्बे', 93: 'त्रियानब्बे', 94: 'चौरानब्बे', 95: 'पन्चानब्बे', 96: 'छयानब्बे', 97: 'सन्तानब्बे', 98: 'अन्ठानब्बे', 99: 'उनन्सय'
}

# --- ENGLISH CONFIG ---
ENGLISH_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ENGLISH_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

def number_to_nepali(n):
    if n < 100: return NEPALI_NUMS[n]
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
    return str(n)

def sanitize_text(text, lang="ne"):
    # 1. Acronyms (2+ Caps letters)
    def replace_acronym(match):
        acronym = match.group(0)
        if lang == "ne":
            return " ".join([ACRONYM_MAP_NE.get(char, char) for char in acronym])
        else:
            # For English, just space them out (TTS -> T T S)
            return " ".join(list(acronym))
    
    text = re.sub(r'\b[A-Z]{2,}\b', replace_acronym, text)
    
    # 2. Numbers (1-6 digits)
    nepali_digits = "०१२३४५६७८९"
    english_digits = "0123456789"
    digit_map = str.maketrans(nepali_digits, english_digits)
    
    def replace_number(match):
        num_str = match.group(0).translate(digit_map)
        try:
            val = int(num_str)
            if 0 <= val <= 999999:
                return number_to_nepali(val) if lang == "ne" else number_to_english(val)
        except:
            pass
        return match.group(0)

    text = re.sub(r'[0-9०-९]{1,6}', replace_number, text)
    return text

if __name__ == "__main__":
    print(sanitize_text("HI my ID is 12345", lang="en"))
    print(sanitize_text("नमस्ते, मेरो आइडी १२३४५६ हो।", lang="ne"))
