#!/usr/bin/env python3
"""
Generate Phase 3a-1 Format Locking Dataset (Combinatorial Version)

Creates diverse Q&A pairs to teach the model:
1. The conversation format (system → user → assistant)
2. When to stop generating (after closing tag)
3. Basic response patterns

Uses combinatorial expansion to generate many unique sequences.
All responses are 1-5 tokens to maximize format exposure per token.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# System prompt - keep consistent for format locking
from core.system_prompts import CONVERSATION_SYSTEM_PROMPT
SYSTEM_PROMPT = CONVERSATION_SYSTEM_PROMPT


def generate_pairs() -> List[Tuple[str, str]]:
    """Generate diverse Q&A pairs with minimal responses using combinatorial expansion."""
    pairs = []
    
    # ==========================================================================
    # COMBINATORIAL COMPONENTS - Define templates and content separately
    # ==========================================================================
    
    # Question templates for single-word answers
    SAY_TEMPLATES = [
        "Say {word}.",
        "Reply with {word}.",
        "Respond {word}.",
        "Answer {word}.",
        "Output {word}.",
        "Just say {word}.",
        "One word: {word}.",
        "Single word: {word}.",
        "Brief: {word}.",
        "Short answer: {word}.",
    ]
    
    # Words for SAY templates
    BASIC_WORDS = [
        "hello", "hi", "goodbye", "bye", "thanks", "please", "sorry", "welcome",
        "done", "complete", "finished", "success", "failed", "error", "correct",
        "wrong", "true", "false", "maybe", "perhaps", "definitely", "absolutely",
        "certainly", "never", "always", "sometimes", "continue", "stop", "start",
        "pause", "resume", "yes", "no", "OK", "confirmed", "denied", "approved",
        "rejected", "accepted", "ready", "waiting", "loading", "processing",
        "saved", "deleted", "updated", "created", "found", "missing", "valid",
        "invalid", "enabled", "disabled", "active", "inactive", "online", "offline",
    ]
    
    # Generate SAY combinations
    for template in SAY_TEMPLATES:
        for word in BASIC_WORDS:
            q = template.format(word=word)
            a = f"{word.capitalize()}."
            pairs.append((q, a))
    
    # ==========================================================================
    # MATH - Combinatorial expansion
    # ==========================================================================
    
    # Addition (limited range to keep dataset manageable)
    for a in range(0, 26):
        for b in range(0, 11):
            pairs.append((f"What is {a} + {b}?", f"{a+b}."))
            pairs.append((f"{a} plus {b}?", f"{a+b}."))
    
    # Subtraction
    for a in range(1, 26):
        for b in range(0, min(a+1, 11)):
            pairs.append((f"What is {a} - {b}?", f"{a-b}."))
            pairs.append((f"{a} minus {b}?", f"{a-b}."))
    
    # Multiplication (times tables)
    for a in range(0, 13):
        for b in range(0, 13):
            pairs.append((f"What is {a} × {b}?", f"{a*b}."))
            pairs.append((f"{a} times {b}?", f"{a*b}."))
    
    # Division (clean results only)
    for a in range(0, 101):
        for b in range(1, 11):
            if a % b == 0:
                pairs.append((f"What is {a} ÷ {b}?", f"{a//b}."))
    
    # ==========================================================================
    # COLORS - Multiple question templates
    # ==========================================================================
    
    COLORS = [
        "red", "blue", "green", "yellow", "orange", "purple", "black", "white",
        "pink", "brown", "gray", "gold", "silver", "cyan", "magenta", "violet",
        "indigo", "turquoise", "maroon", "navy", "teal", "olive", "coral", "lime",
    ]
    
    COLOR_TEMPLATES = [
        ("Name a color: {color}.", "{Color}."),
        ("Say {color}.", "{Color}."),
        ("What color is {color}?", "{Color}."),
        ("Color: {color}.", "{Color}."),
    ]
    
    for template_q, template_a in COLOR_TEMPLATES:
        for color in COLORS:
            q = template_q.format(color=color)
            a = template_a.format(Color=color.capitalize())
            pairs.append((q, a))
    
    # ==========================================================================
    # DAYS & MONTHS - Multiple templates
    # ==========================================================================
    
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    DAY_TEMPLATES = [
        ("Say {day}.", "{day}."),
        ("What day is {day}?", "{day}."),
        ("Day: {day}.", "{day}."),
    ]
    
    for template_q, template_a in DAY_TEMPLATES:
        for day in DAYS:
            pairs.append((template_q.format(day=day), template_a.format(day=day)))
    
    for i, day in enumerate(DAYS):
        pairs.append((f"What day comes after {DAYS[i-1]}?", f"{day}."))
        pairs.append((f"Day after {DAYS[i-1]}?", f"{day}."))
        pairs.append((f"Day number {i+1}?", f"{day}."))
    
    MONTHS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    
    MONTH_TEMPLATES = [
        ("Say {month}.", "{month}."),
        ("Month: {month}.", "{month}."),
    ]
    
    for template_q, template_a in MONTH_TEMPLATES:
        for month in MONTHS:
            pairs.append((template_q.format(month=month), template_a.format(month=month)))
    
    for i, month in enumerate(MONTHS):
        pairs.append((f"What month is number {i+1}?", f"{month}."))
        pairs.append((f"Month {i+1}?", f"{month}."))
        pairs.append((f"Month after {MONTHS[i-1]}?", f"{month}."))
    
    # ==========================================================================
    # CAPITALS - Expanded with templates
    # ==========================================================================
    
    CAPITALS = {
        "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo", "Italy": "Rome",
        "Spain": "Madrid", "UK": "London", "USA": "Washington", "China": "Beijing",
        "Russia": "Moscow", "Brazil": "Brasília", "Australia": "Canberra",
        "Canada": "Ottawa", "India": "New Delhi", "Mexico": "Mexico City",
        "Egypt": "Cairo", "Poland": "Warsaw", "Netherlands": "Amsterdam",
        "Belgium": "Brussels", "Austria": "Vienna", "Switzerland": "Bern",
        "Sweden": "Stockholm", "Norway": "Oslo", "Denmark": "Copenhagen",
        "Finland": "Helsinki", "Greece": "Athens", "Portugal": "Lisbon",
        "Ireland": "Dublin", "Turkey": "Ankara", "Thailand": "Bangkok",
        "Vietnam": "Hanoi", "Indonesia": "Jakarta", "South Korea": "Seoul",
        "Argentina": "Buenos Aires", "Chile": "Santiago", "Peru": "Lima",
    }
    
    CAPITAL_TEMPLATES = [
        "Capital of {country}?",
        "What is the capital of {country}?",
        "{country} capital?",
    ]
    
    for country, capital in CAPITALS.items():
        for template in CAPITAL_TEMPLATES:
            pairs.append((template.format(country=country), f"{capital}."))
    
    # ==========================================================================
    # OPPOSITES - Expanded
    # ==========================================================================
    
    OPPOSITES = {
        "hot": "Cold", "big": "Small", "fast": "Slow", "light": "Dark",
        "good": "Bad", "happy": "Sad", "old": "Young", "rich": "Poor",
        "easy": "Hard", "open": "Closed", "full": "Empty", "wet": "Dry",
        "loud": "Quiet", "strong": "Weak", "tall": "Short", "wide": "Narrow",
        "thick": "Thin", "heavy": "Light", "soft": "Hard", "smooth": "Rough",
        "clean": "Dirty", "safe": "Dangerous", "new": "Old", "early": "Late",
        "high": "Low", "near": "Far", "long": "Short", "deep": "Shallow",
        "bright": "Dim", "sharp": "Dull", "sweet": "Sour", "warm": "Cool",
    }
    
    OPPOSITE_TEMPLATES = [
        "Opposite of {word}?",
        "What is the opposite of {word}?",
        "Antonym of {word}?",
        "What is the inverse of {word}?",
    ]
    
    for word, opposite in OPPOSITES.items():
        for template in OPPOSITE_TEMPLATES:
            pairs.append((template.format(word=word), f"{opposite}."))
    
    # ==========================================================================
    # YES/NO QUESTIONS - Expanded
    # ==========================================================================
    
    YES_FACTS = [
        "Is water wet?", "Is the sky blue?", "Is grass green?", "Is 2+2=4?",
        "Is Earth round?", "Is Python a language?", "Is Java a language?",
        "Is 10 > 5?", "Is Monday a day?", "Is 1 odd?", "Is 2 even?",
        "Is red a color?", "Is gold a metal?", "Is ice cold?", "Is fire hot?",
        "Is the sun a star?", "Is water H2O?", "Is 100 > 50?", "Is 0 even?",
        "Is December a month?", "Is Sunday a day?", "Is oxygen a gas?",
        "Is wood flammable?", "Is iron magnetic?", "Is glass transparent?",
        "Is sugar sweet?", "Is lemon sour?", "Is snow white?", "Is night dark?",
        "Is summer warm?", "Is winter cold?", "Is spring a season?",
        "Is 7 a prime?", "Is 12 divisible by 3?", "Is 15 divisible by 5?",
    ]
    
    NO_FACTS = [
        "Is fire cold?", "Is ice hot?", "Is 2+2=5?", "Is Earth flat?",
        "Is 3 > 7?", "Is 0 positive?", "Is silence loud?", "Is night bright?",
        "Is water dry?", "Is the moon a star?", "Is 1 even?", "Is 3 even?",
        "Is snow black?", "Is coal white?", "Is summer cold?", "Is winter hot?",
        "Is 10 < 5?", "Is 100 < 50?", "Is a fish a mammal?", "Is a snake a mammal?",
        "Is glass opaque?", "Is wood transparent?", "Is helium heavy?",
        "Is lead light?", "Is sugar salty?", "Is vinegar sweet?",
        "Is 9 a prime?", "Is 10 divisible by 3?", "Is 7 divisible by 2?",
    ]
    
    for fact in YES_FACTS:
        pairs.append((fact, "Yes."))
    for fact in NO_FACTS:
        pairs.append((fact, "No."))
    
    # ==========================================================================
    # LETTERS & NUMBERS - Templates
    # ==========================================================================
    
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    LETTER_TEMPLATES = [
        ("Letter number {num}?", "{letter}."),
        ("What is letter {num}?", "{letter}."),
        ("Say letter {letter}.", "{letter}."),
        ("Letter: {letter}.", "{letter}."),
        ("Repeat following letter: {letter}.", "{letter}."),
        ("Repeat the letter: {letter}.", "{letter}."),
    ]
    
    for i, letter in enumerate(ALPHABET):
        for template_q, template_a in LETTER_TEMPLATES:
            q = template_q.format(num=i+1, letter=letter)
            a = template_a.format(letter=letter)
            pairs.append((q, a))
    
    # Number words
    NUMBER_WORDS = {
        0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
        6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
        11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen",
        16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty",
        30: "Thirty", 40: "Forty", 50: "Fifty", 100: "Hundred", 1000: "Thousand",
        1000000: "Million", 1000000000: "Billion", 1000000000000: "Trillion",
        111: "One hundred eleven", 222: "Two hundred twenty-two", 333: "Three hundred thirty-three",
        444: "Four hundred forty-four", 555: "Five hundred fifty-five", 666: "Six hundred sixty-six",
        777: "Seven hundred seventy-seven", 888: "Eight hundred eighty-eight", 999: "Nine hundred ninety-nine",
    }
    
    NUMBER_TEMPLATES = [
        ("{num} in words?", "{word}."),
        ("Spell {num}.", "{word}."),
        ("Say {num} as a word.", "{word}."),
        ("Repeat the number in words: {num}.", "{word}."),
    ]
    
    for num, word in NUMBER_WORDS.items():
        for template_q, template_a in NUMBER_TEMPLATES:
            pairs.append((template_q.format(num=num), template_a.format(word=word)))
    
    # ==========================================================================
    # ANIMALS - Expanded
    # ==========================================================================
    
    ANIMAL_SOUNDS = {
        "dog": "Bark", "cat": "Meow", "cow": "Moo", "pig": "Oink",
        "duck": "Quack", "bird": "Chirp", "lion": "Roar", "snake": "Hiss",
        "bee": "Buzz", "wolf": "Howl", "frog": "Croak", "owl": "Hoot",
        "horse": "Neigh", "sheep": "Baa", "rooster": "Crow", "crow": "Caw",
        "donkey": "Bray", "goat": "Bleat", "mouse": "Squeak",
    }
    
    SOUND_TEMPLATES = [
        "{animal} sound?",
        "What sound does a {animal} make?",
        "Sound of {animal}?",
    ]
    
    for animal, sound in ANIMAL_SOUNDS.items():
        for template in SOUND_TEMPLATES:
            q = template.format(animal=animal)
            pairs.append((q, f"{sound}."))
    
    ANIMAL_FACTS = [
        ("Largest animal?", "Blue whale."),
        ("Fastest animal?", "Cheetah."),
        ("Tallest animal?", "Giraffe."),
        ("King of jungle?", "Lion."),
        ("Man's best friend?", "Dog."),
        ("Largest land animal?", "Elephant."),
        ("Fastest bird?", "Falcon."),
        ("Largest bird?", "Ostrich."),
    ]
    pairs.extend(ANIMAL_FACTS)
    
    # ==========================================================================
    # DIRECTIONS - Expanded
    # ==========================================================================
    
    DIRECTIONS = [
        "up", "down", "left", "right", "north", "south", "east", "west",
        "forward", "backward", "inside", "outside", "above", "below",
        "here", "there", "near", "far", "front", "back",
    ]
    
    DIRECTION_TEMPLATES = [
        ("Which way is {dir}?", "{Dir}."),
        ("Say {dir}.", "{Dir}."),
        ("Direction: {dir}.", "{Dir}."),
    ]
    
    for direction in DIRECTIONS:
        for template_q, template_a in DIRECTION_TEMPLATES:
            q = template_q.format(dir=direction)
            a = template_a.format(Dir=direction.capitalize())
            pairs.append((q, a))
    
    # ==========================================================================
    # GREETINGS - Expanded
    # ==========================================================================
    
    GREETINGS = [
        ("Good morning!", "Good morning!"),
        ("Good afternoon!", "Good afternoon!"),
        ("Good evening!", "Good evening!"),
        ("Good night!", "Good night!"),
        ("Hello!", "Hello!"),
        ("Hi!", "Hi!"),
        ("Hey!", "Hey!"),
        ("Greetings!", "Greetings!"),
        ("Howdy!", "Howdy!"),
        ("What's up?", "Hello!"),
        ("How are you?", "I'm good, thank you!"),
        ("How are you doing?", "I'm good, thank you!"),
        ("Hola!", "Hola!"),
        ("Bonjour!", "Bonjour!"),
        ("Guten Tag!", "Guten Tag!"),
        ("Ciao!", "Ciao!"),
        ("Welcome!", "Welcome!"),
    ]
    pairs.extend(GREETINGS)
    
    # ==========================================================================
    # STATUS & ACKNOWLEDGMENTS - Expanded
    # ==========================================================================
    
    STATUS_QA = [
        ("Status?", "OK."),
        ("How are you?", "Good."),
        ("All good?", "Yes."),
        ("Everything OK?", "Yes."),
        ("Ready to help?", "Yes."),
        ("Can you hear me?", "Yes."),
        ("Are you there?", "Yes."),
        ("Still there?", "Yes."),
        ("Working?", "Yes."),
        ("Online?", "Yes."),
        ("Ready?", "Ready."),
        ("Understood?", "Understood."),
        ("Clear?", "Clear."),
        ("Confirm?", "Confirmed."),
        ("Acknowledge?", "Acknowledged."),
        ("Copy?", "Copy."),
        ("Roger?", "Roger."),
        ("Got it?", "Got it."),
        ("Who are you?", "MyPT."),
        ("What is your name?", "MyPT."),

    ]
    pairs.extend(STATUS_QA)
    
    # ==========================================================================
    # PROGRAMMING & TECH - Expanded
    # ==========================================================================
    
    FILE_EXTENSIONS = {
        "Python": ".py", "JavaScript": ".js", "TypeScript": ".ts",
        "HTML": ".html", "CSS": ".css", "JSON": ".json", "YAML": ".yaml",
        "Markdown": ".md", "Text": ".txt", "PNG": ".png", "JPEG": ".jpg",
        "GIF": ".gif", "PDF": ".pdf", "XML": ".xml", "CSV": ".csv",
        "Java": ".java", "C++": ".cpp", "C": ".c", "Ruby": ".rb",
        "Go": ".go", "Rust": ".rs", "PHP": ".php", "SQL": ".sql",
        "Word": ".docx", "Excel": ".xlsx", "PowerPoint": ".pptx",
    }
    
    EXT_TEMPLATES = [
        "{lang} file extension?",
        "{lang} extension?",
        "Extension for {lang}?",
        "Extension of {lang}?",
        "What is the extension of {lang}?",

    ]
    
    for lang, ext in FILE_EXTENSIONS.items():
        for template in EXT_TEMPLATES:
            pairs.append((template.format(lang=lang), ext))
    
    SYMBOLS = [
        ("Plus sign?", "+"), ("Minus sign?", "-"), ("Multiply sign?", "*"),
        ("Divide sign?", "/"), ("Equals sign?", "="), ("Greater than?", ">"),
        ("Less than?", "<"), ("Not equal?", "!="), ("Modulo sign?", "%"),
        ("At sign?", "@"), ("Hash sign?", "#"), ("Dollar sign?", "$"),
        ("Percent sign?", "%"), ("Ampersand?", "&"), ("Asterisk?", "*"),
        ("Exclamation mark?", "!"), ("Question mark?", "?"), ("Pipe?", "|"),
        ("Backslash?", "\\"), ("Slash?", "/"), ("Colon?", ":"),
        ("Semicolon?", ";"), ("Comma?", ","), ("Period?", "."),
        ("Quote?", "'"), ("Double quote?", '"'), ("Left parenthesis?", "("),
        ("Right parenthesis?", ")"), ("Left bracket?", "["),
        ("Right bracket?", "]"), ("Left brace?", "{"), ("Right brace?", "}"),
        ("Left angle bracket?", "<"), ("Right angle bracket?", ">"),
        ("Left curly brace?", "{"), ("Right curly brace?", "}"),
        ("Left square bracket?", "["), ("Right square bracket?", "]"),
        ("Left parenthesis?", "("), ("Right parenthesis?", ")"),
        ("Left curly brace?", "{"), ("Right curly brace?", "}"),
        ("Left square bracket?", "["), ("Right square bracket?", "]"),
        ("Left parenthesis?", "("), ("Right parenthesis?", ")"),
    ]
    pairs.extend(SYMBOLS)
    
    UNITS = [
        ("Meters abbreviation?", "m"), ("Kilometers abbreviation?", "km"),
        ("Centimeters abbreviation?", "cm"), ("Kilograms abbreviation?", "kg"),
        ("Grams abbreviation?", "g"), ("Seconds abbreviation?", "s"),
        ("Minutes abbreviation?", "min"), ("Hours abbreviation?", "h"),
        ("Celsius abbreviation?", "°C"), ("Liters abbreviation?", "L"),
        ("Bytes abbreviation?", "B"), ("Kilobytes abbreviation?", "KB"),
        ("Megabytes abbreviation?", "MB"), ("Gigabytes abbreviation?", "GB"),
    ]
    pairs.extend(UNITS)
    
    # ==========================================================================
    # PROGRAMMING CONCEPTS - Syntax, Keywords, Data Types
    # ==========================================================================
    
    # Print/Output statements by language
    PRINT_STATEMENTS = {
        "Python": "print()", "JavaScript": "console.log()", "Java": "System.out.println()",
        "C": "printf()", "C++": "cout <<", "C#": "Console.WriteLine()",
        "Ruby": "puts", "Go": "fmt.Println()", "Rust": "println!()",
        "PHP": "echo", "Swift": "print()", "Kotlin": "println()",
    }
    
    PRINT_TEMPLATES = [
        "Print in {lang}?",
        "How to print in {lang}?",
        "{lang} print statement?",
        "Output in {lang}?",
    ]
    
    for lang, stmt in PRINT_STATEMENTS.items():
        for template in PRINT_TEMPLATES:
            pairs.append((template.format(lang=lang), stmt))
    
    # Comment syntax by language
    COMMENTS = {
        "Python": "#", "JavaScript": "//", "Java": "//", "C": "//",
        "C++": "//", "Ruby": "#", "Go": "//", "Rust": "//",
        "PHP": "//", "Shell": "#", "Bash": "#", "SQL": "--",
        "HTML": "<!-- -->", "CSS": "/* */", "Lua": "--",
    }
    
    COMMENT_TEMPLATES = [
        "Comment in {lang}?",
        "{lang} comment syntax?",
        "How to comment in {lang}?",
        "Single line comment in {lang}?",
    ]
    
    for lang, syntax in COMMENTS.items():
        for template in COMMENT_TEMPLATES:
            pairs.append((template.format(lang=lang), syntax))
    
    # Boolean values by language
    BOOLEANS_TRUE = {
        "Python": "True", "JavaScript": "true", "Java": "true",
        "C": "1", "C++": "true", "Ruby": "true", "Go": "true",
        "Rust": "true", "PHP": "true", "SQL": "TRUE",
    }
    
    BOOLEANS_FALSE = {
        "Python": "False", "JavaScript": "false", "Java": "false",
        "C": "0", "C++": "false", "Ruby": "false", "Go": "false",
        "Rust": "false", "PHP": "false", "SQL": "FALSE",
    }
    
    for lang, val in BOOLEANS_TRUE.items():
        pairs.append((f"True in {lang}?", val))
        pairs.append((f"{lang} true value?", val))
    
    for lang, val in BOOLEANS_FALSE.items():
        pairs.append((f"False in {lang}?", val))
        pairs.append((f"{lang} false value?", val))
    
    # Null/None/Nil values by language
    NULL_VALUES = {
        "Python": "None", "JavaScript": "null", "Java": "null",
        "C": "NULL", "C++": "nullptr", "Ruby": "nil", "Go": "nil",
        "Rust": "None", "PHP": "null", "SQL": "NULL", "Swift": "nil",
    }
    
    NULL_TEMPLATES = [
        "Null in {lang}?",
        "{lang} null value?",
        "Empty/null in {lang}?",
    ]
    
    for lang, val in NULL_VALUES.items():
        for template in NULL_TEMPLATES:
            pairs.append((template.format(lang=lang), val))
    
    # Array/List syntax by language
    ARRAY_SYNTAX = {
        "Python": "[]", "JavaScript": "[]", "Java": "new int[]{}",
        "C": "int arr[]", "Ruby": "[]", "Go": "[]int{}",
        "PHP": "array()", "Swift": "[]", "Kotlin": "arrayOf()",
    }
    
    for lang, syntax in ARRAY_SYNTAX.items():
        pairs.append((f"Array in {lang}?", syntax))
        pairs.append((f"{lang} array syntax?", syntax))
    
    # Dictionary/Map/Object syntax
    DICT_SYNTAX = {
        "Python": "{}", "JavaScript": "{}", "Java": "HashMap<>",
        "Ruby": "{}", "Go": "map[]", "PHP": "array()",
    }
    
    for lang, syntax in DICT_SYNTAX.items():
        pairs.append((f"Dictionary in {lang}?", syntax))
        pairs.append((f"{lang} dict syntax?", syntax))
    
    # Function definition keywords
    FUNC_KEYWORDS = {
        "Python": "def", "JavaScript": "function", "Java": "void/type",
        "C": "void/type", "Ruby": "def", "Go": "func", "Rust": "fn",
        "PHP": "function", "Swift": "func", "Kotlin": "fun",
    }
    
    for lang, kw in FUNC_KEYWORDS.items():
        pairs.append((f"Function keyword in {lang}?", kw))
        pairs.append((f"{lang} function keyword?", kw))
    
    # String quotes
    STRING_QUOTES = [
        ("String in Python?", '""'), ("String in JavaScript?", '""'),
        ("String in Java?", '""'), ("String in C?", '""'),
        ("Char in C?", "''"), ("Char in Java?", "''"),
        ("Raw string in Python?", 'r""'), ("Template string in JS?", "``"),
        ("F-string in Python?", 'f""'),
    ]
    pairs.extend(STRING_QUOTES)
    
    # HTTP Methods
    HTTP_METHODS = [
        ("HTTP method for read?", "GET"),
        ("HTTP method for create?", "POST"),
        ("HTTP method for update?", "PUT"),
        ("HTTP method for delete?", "DELETE"),
        ("HTTP method for partial update?", "PATCH"),
        ("HTTP method for headers only?", "HEAD"),
        ("HTTP method for options?", "OPTIONS"),
    ]
    pairs.extend(HTTP_METHODS)
    
    # HTTP Status Codes
    HTTP_CODES = [
        ("HTTP code for OK?", "200"), ("HTTP code for created?", "201"),
        ("HTTP code for no content?", "204"), ("HTTP code for redirect?", "301"),
        ("HTTP code for bad request?", "400"), ("HTTP code for unauthorized?", "401"),
        ("HTTP code for forbidden?", "403"), ("HTTP code for not found?", "404"),
        ("HTTP code for server error?", "500"), ("HTTP code for bad gateway?", "502"),
    ]
    pairs.extend(HTTP_CODES)
    
    # Data types
    DATA_TYPES = [
        ("Integer type in Python?", "int"), ("Float type in Python?", "float"),
        ("String type in Python?", "str"), ("Boolean type in Python?", "bool"),
        ("List type in Python?", "list"), ("Dict type in Python?", "dict"),
        ("Number type in JavaScript?", "number"), ("String type in JavaScript?", "string"),
        ("Boolean type in JavaScript?", "boolean"), ("Array type in JavaScript?", "array"),
        ("Integer type in Java?", "int"), ("Double type in Java?", "double"),
        ("String type in Java?", "String"), ("Boolean type in Java?", "boolean"),
    ]
    pairs.extend(DATA_TYPES)
    
    # Common programming keywords
    KEYWORDS = [
        ("Loop keyword for iteration?", "for"), ("Loop keyword for condition?", "while"),
        ("Conditional keyword?", "if"), ("Alternative keyword?", "else"),
        ("Return keyword?", "return"), ("Break loop keyword?", "break"),
        ("Skip iteration keyword?", "continue"), ("Import keyword in Python?", "import"),
        ("Import keyword in Java?", "import"), ("Import keyword in JavaScript?", "import"),
        ("Class keyword?", "class"), ("Try block keyword?", "try"),
        ("Catch block keyword?", "catch"), ("Except block in Python?", "except"),
        ("Finally block keyword?", "finally"), ("Throw keyword?", "throw"),
        ("Raise keyword in Python?", "raise"), ("Assert keyword?", "assert"),
        ("Lambda keyword in Python?", "lambda"), ("Arrow function in JS?", "=>"),
    ]
    pairs.extend(KEYWORDS)
    
    # Common operators
    OPERATORS = [
        ("Equality operator?", "=="), ("Strict equality in JS?", "==="),
        ("Not equal operator?", "!="), ("Strict not equal in JS?", "!=="),
        ("And operator in Python?", "and"), ("Or operator in Python?", "or"),
        ("Not operator in Python?", "not"), ("And operator in JS?", "&&"),
        ("Or operator in JS?", "||"), ("Not operator in JS?", "!"),
        ("Increment operator?", "++"), ("Decrement operator?", "--"),
        ("Add assign operator?", "+="), ("Subtract assign operator?", "-="),
        ("Multiply assign operator?", "*="), ("Divide assign operator?", "/="),
        ("Floor division in Python?", "//"), ("Exponent operator in Python?", "**"),
        ("Ternary operator in JS?", "? :"), ("Null coalescing in JS?", "??"),
    ]
    pairs.extend(OPERATORS)
    
    # Git commands
    GIT_COMMANDS = [
        ("Git clone command?", "git clone"), ("Git pull command?", "git pull"),
        ("Git push command?", "git push"), ("Git commit command?", "git commit"),
        ("Git add command?", "git add"), ("Git status command?", "git status"),
        ("Git branch command?", "git branch"), ("Git checkout command?", "git checkout"),
        ("Git merge command?", "git merge"), ("Git log command?", "git log"),
        ("Git diff command?", "git diff"), ("Git stash command?", "git stash"),
    ]
    pairs.extend(GIT_COMMANDS)
    
    # Common CLI commands
    CLI_COMMANDS = [
        ("List files command?", "ls"), ("Change directory?", "cd"),
        ("Print working directory?", "pwd"), ("Make directory?", "mkdir"),
        ("Remove file?", "rm"), ("Copy file?", "cp"), ("Move file?", "mv"),
        ("View file contents?", "cat"), ("Search in files?", "grep"),
        ("Find files?", "find"), ("Download URL?", "curl"),
        ("Package manager for Python?", "pip"), ("Package manager for Node?", "npm"),
        ("Package manager for Ruby?", "gem"), ("Package manager for Rust?", "cargo"),
    ]
    pairs.extend(CLI_COMMANDS)
    
    # ==========================================================================
    # COMPARISONS - Expanded  
    # ==========================================================================
    
    COMPARISONS = [
        ("Bigger: ant or elephant?", "Elephant."),
        ("Faster: car or bicycle?", "Car."),
        ("Hotter: sun or ice?", "Sun."),
        ("Taller: tree or grass?", "Tree."),
        ("Heavier: feather or rock?", "Rock."),
        ("Older: parent or child?", "Parent."),
        ("Longer: year or day?", "Year."),
        ("More: dozen or ten?", "Dozen."),
        ("Brighter: day or night?", "Day."),
        ("Wetter: ocean or desert?", "Ocean."),
        ("Bigger: whale or mouse?", "Whale."),
        ("Faster: plane or train?", "Plane."),
        ("Colder: ice or fire?", "Ice."),
        ("Deeper: ocean or lake?", "Ocean."),
        ("Louder: whisper or shout?", "Shout."),
        ("Harder: diamond or glass?", "Diamond."),
        ("Higher: mountain or hill?", "Mountain."),
        ("Faster: light or sound?", "Light."),
        ("Bigger: Earth or Moon?", "Earth."),
        ("Larger: Jupiter or Mars?", "Jupiter."),
    ]
    pairs.extend(COMPARISONS)
    
    # ==========================================================================
    # POLITE RESPONSES - Expanded
    # ==========================================================================
    
    POLITE = [
        ("Thank you.", "You're welcome."),
        ("Thanks!", "You're welcome!"),
        ("Thanks a lot.", "You're welcome."),
        ("I appreciate it.", "You're welcome."),
        ("Sorry.", "No problem."),
        ("My apologies.", "No worries."),
        ("Excuse me.", "Of course."),
        ("Please help.", "Sure."),
        ("Can you help?", "Yes."),
        ("Will you help?", "Yes."),
        ("Help me.", "OK."),
        ("Please.", "OK."),
        ("Would you mind?", "Not at all."),
        ("Is that OK?", "Yes."),
        ("May I?", "Yes."),
    ]
    pairs.extend(POLITE)
    
    # ==========================================================================
    # GERMAN CONTENT - Basic German Q&A
    # ==========================================================================
    
    GERMAN_BASIC = [
        ("Sag Hallo.", "Hallo."),
        ("Sag Ja.", "Ja."),
        ("Sag Nein.", "Nein."),
        ("Sag Danke.", "Danke."),
        ("Sag Bitte.", "Bitte."),
        ("Sag OK.", "OK."),
        ("Guten Morgen!", "Guten Morgen!"),
        ("Guten Tag!", "Guten Tag!"),
        ("Guten Abend!", "Guten Abend!"),
        ("Gute Nacht!", "Gute Nacht!"),
        ("Wie geht's?", "Gut."),
        ("Alles klar?", "Ja."),
        ("Verstanden?", "Verstanden."),
        ("Fertig?", "Fertig."),
        ("Bereit?", "Bereit."),
        ("Hauptstadt von Deutschland?", "Berlin."),
        ("Hauptstadt von Frankreich?", "Paris."),
        ("Hauptstadt von Italien?", "Rom."),
        ("Hauptstadt von Spanien?", "Madrid."),
        ("Was ist 1 + 1?", "2."),
        ("Was ist 2 + 2?", "4."),
        ("Was ist 5 + 5?", "10."),
        ("Was ist 10 - 5?", "5."),
        ("Ist Wasser nass?", "Ja."),
        ("Ist Feuer kalt?", "Nein."),
        ("Ist der Himmel blau?", "Ja."),
        ("Gegenteil von heiß?", "Kalt."),
        ("Gegenteil von groß?", "Klein."),
        ("Gegenteil von schnell?", "Langsam."),
        ("Welche Farbe: rot?", "Rot."),
        ("Welche Farbe: blau?", "Blau."),
        ("Welche Farbe: grün?", "Grün."),
        ("Hund Geräusch?", "Bellen."),
        ("Katze Geräusch?", "Miau."),
    ]
    pairs.extend(GERMAN_BASIC)
    
    return pairs


def create_episode(question: str, answer: str, episode_id: int) -> dict:
    """Create a single episode in the expected format."""
    # Detect language from content
    german_indicators = ["Sag ", "Guten ", "Gute ", "Hauptstadt von", "Was ist", "Ist ", 
                         "Gegenteil von", "Welche Farbe", "Geräusch", "Verstanden", "Fertig", "Bereit"]
    is_german = any(indicator in question for indicator in german_indicators)
    
    return {
        "system": SYSTEM_PROMPT,
        "context": f"episode_id: format_lock_{episode_id:04d}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "language": "de" if is_german else "en"
    }


def main():
    output_dir = Path("data/sft_format_lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mypt_format_lock_v1.jsonl"
    
    # Generate pairs
    pairs = generate_pairs()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    pairs = unique_pairs
    
    # Shuffle for variety
    random.seed(42)
    random.shuffle(pairs)
    
    # Create episodes
    episodes = []
    for i, (q, a) in enumerate(pairs):
        episode = create_episode(q, a, i)
        episodes.append(episode)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    # Stats
    total_pairs = len(pairs)
    avg_q_len = sum(len(q) for q, _ in pairs) / total_pairs
    avg_a_len = sum(len(a) for _, a in pairs) / total_pairs
    
    # Count by language
    en_count = sum(1 for e in episodes if e["language"] == "en")
    de_count = sum(1 for e in episodes if e["language"] == "de")
    
    print(f"✅ Generated {total_pairs} unique Q&A pairs")
    print(f"   Output: {output_file}")
    print(f"   English: {en_count}, German: {de_count}")
    print(f"   Average question length: {avg_q_len:.1f} chars")
    print(f"   Average answer length: {avg_a_len:.1f} chars")
    print()
    print("Sample pairs:")
    for i in range(min(10, len(pairs))):
        q, a = pairs[i]
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()


if __name__ == "__main__":
    main()
