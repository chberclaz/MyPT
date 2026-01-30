#!/usr/bin/env python3
"""
Generate Phase 3a-1 Format Locking Dataset

Creates minimal Q&A pairs to teach the model:
1. The conversation format (system → user → assistant)
2. When to stop generating (after closing tag)
3. Basic response patterns

All responses are 1-5 tokens to maximize format exposure per token.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

# System prompt - keep consistent for format locking
SYSTEM_PROMPT = "You are MyPT, a helpful assistant."

def generate_pairs() -> List[Tuple[str, str]]:
    """Generate diverse Q&A pairs with minimal responses."""
    pairs = []
    
    # === ACKNOWLEDGMENTS (high frequency - most basic pattern) ===
    ack_questions = [
        "Say OK.", "Respond with OK.", "Just say OK.", "Reply OK.",
        "Say yes.", "Respond yes.", "Answer yes.", "Just yes.",
        "Say no.", "Respond no.", "Answer no.", "Just no.",
        "Acknowledge.", "Confirm.", "Understood?", "Got it?",
        "Ready?", "Clear?", "Agreed?", "Accept?",
    ]
    ack_answers = {
        "Say OK.": "OK.", "Respond with OK.": "OK.", "Just say OK.": "OK.", "Reply OK.": "OK.",
        "Say yes.": "Yes.", "Respond yes.": "Yes.", "Answer yes.": "Yes.", "Just yes.": "Yes.",
        "Say no.": "No.", "Respond no.": "No.", "Answer no.": "No.", "Just no.": "No.",
        "Acknowledge.": "Acknowledged.", "Confirm.": "Confirmed.", "Understood?": "Understood.",
        "Got it?": "Got it.", "Ready?": "Ready.", "Clear?": "Clear.", "Agreed?": "Agreed.",
        "Accept?": "Accepted.",
    }
    for q, a in ack_answers.items():
        pairs.append((q, a))
    
    # === SINGLE WORDS ===
    word_templates = [
        ("Say hello.", "Hello."),
        ("Say hi.", "Hi."),
        ("Say goodbye.", "Goodbye."),
        ("Say thanks.", "Thanks."),
        ("Say please.", "Please."),
        ("Say sorry.", "Sorry."),
        ("Say welcome.", "Welcome."),
        ("Say done.", "Done."),
        ("Say complete.", "Complete."),
        ("Say finished.", "Finished."),
        ("Say success.", "Success."),
        ("Say failed.", "Failed."),
        ("Say error.", "Error."),
        ("Say correct.", "Correct."),
        ("Say wrong.", "Wrong."),
        ("Say true.", "True."),
        ("Say false.", "False."),
        ("Say maybe.", "Maybe."),
        ("Say perhaps.", "Perhaps."),
        ("Say definitely.", "Definitely."),
        ("Say absolutely.", "Absolutely."),
        ("Say certainly.", "Certainly."),
        ("Say never.", "Never."),
        ("Say always.", "Always."),
        ("Say sometimes.", "Sometimes."),
        ("Say continue.", "Continue."),
        ("Say stop.", "Stop."),
        ("Say start.", "Start."),
        ("Say pause.", "Pause."),
        ("Say resume.", "Resume."),
    ]
    pairs.extend(word_templates)
    
    # === NUMBERS ===
    for i in range(1, 51):
        pairs.append((f"What is {i-1} plus 1?", f"{i}."))
    for i in range(2, 21):
        pairs.append((f"What is {i*2} divided by 2?", f"{i}."))
    for i in range(1, 11):
        pairs.append((f"What is {i} times 1?", f"{i}."))
    
    number_words = [
        ("One word: one.", "One."),
        ("One word: two.", "Two."),
        ("One word: three.", "Three."),
        ("One word: four.", "Four."),
        ("One word: five.", "Five."),
        ("One word: ten.", "Ten."),
        ("One word: zero.", "Zero."),
        ("One word: hundred.", "Hundred."),
        ("One word: thousand.", "Thousand."),
        ("One word: million.", "Million."),
    ]
    pairs.extend(number_words)
    
    # === COLORS ===
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown", "gray", "gold", "silver"]
    for color in colors:
        pairs.append((f"Name a color: {color}.", f"{color.capitalize()}."))
        pairs.append((f"Say {color}.", f"{color.capitalize()}."))
    
    # === DAYS/MONTHS ===
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i, day in enumerate(days):
        pairs.append((f"What day comes after {days[i-1]}?", f"{day}."))
        pairs.append((f"Say {day}.", f"{day}."))
    
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    for i, month in enumerate(months):
        pairs.append((f"What month is number {i+1}?", f"{month}."))
        pairs.append((f"Say {month}.", f"{month}."))
    
    # === DIRECTIONS ===
    directions = [
        ("Which way is up?", "Up."),
        ("Which way is down?", "Down."),
        ("Which way is left?", "Left."),
        ("Which way is right?", "Right."),
        ("Which way is north?", "North."),
        ("Which way is south?", "South."),
        ("Which way is east?", "East."),
        ("Which way is west?", "West."),
        ("Say forward.", "Forward."),
        ("Say backward.", "Backward."),
        ("Say inside.", "Inside."),
        ("Say outside.", "Outside."),
        ("Say above.", "Above."),
        ("Say below.", "Below."),
        ("Say here.", "Here."),
        ("Say there.", "There."),
    ]
    pairs.extend(directions)
    
    # === SIMPLE FACTS (1-2 word answers) ===
    facts = [
        ("Capital of France?", "Paris."),
        ("Capital of Germany?", "Berlin."),
        ("Capital of Japan?", "Tokyo."),
        ("Capital of Italy?", "Rome."),
        ("Capital of Spain?", "Madrid."),
        ("Capital of UK?", "London."),
        ("Capital of USA?", "Washington."),
        ("Capital of China?", "Beijing."),
        ("Capital of Russia?", "Moscow."),
        ("Capital of Brazil?", "Brasília."),
        ("Capital of Australia?", "Canberra."),
        ("Capital of Canada?", "Ottawa."),
        ("Capital of India?", "New Delhi."),
        ("Capital of Mexico?", "Mexico City."),
        ("Capital of Egypt?", "Cairo."),
        ("Largest ocean?", "Pacific."),
        ("Largest continent?", "Asia."),
        ("Smallest continent?", "Australia."),
        ("Longest river?", "Nile."),
        ("Highest mountain?", "Everest."),
        ("Largest planet?", "Jupiter."),
        ("Smallest planet?", "Mercury."),
        ("Closest star?", "Sun."),
        ("Earth's satellite?", "Moon."),
        ("Frozen water?", "Ice."),
        ("H2O is?", "Water."),
        ("Opposite of hot?", "Cold."),
        ("Opposite of big?", "Small."),
        ("Opposite of fast?", "Slow."),
        ("Opposite of light?", "Dark."),
        ("Opposite of good?", "Bad."),
        ("Opposite of happy?", "Sad."),
        ("Opposite of old?", "Young."),
        ("Opposite of rich?", "Poor."),
        ("Opposite of easy?", "Hard."),
        ("Opposite of open?", "Closed."),
        ("Opposite of full?", "Empty."),
        ("Opposite of wet?", "Dry."),
        ("Opposite of loud?", "Quiet."),
        ("Opposite of strong?", "Weak."),
    ]
    pairs.extend(facts)
    
    # === ANIMALS ===
    animals = [
        ("Dog sound?", "Bark."),
        ("Cat sound?", "Meow."),
        ("Cow sound?", "Moo."),
        ("Pig sound?", "Oink."),
        ("Duck sound?", "Quack."),
        ("Bird sound?", "Chirp."),
        ("Lion sound?", "Roar."),
        ("Snake sound?", "Hiss."),
        ("Bee sound?", "Buzz."),
        ("Wolf sound?", "Howl."),
        ("Largest animal?", "Blue whale."),
        ("Fastest animal?", "Cheetah."),
        ("Tallest animal?", "Giraffe."),
        ("King of jungle?", "Lion."),
        ("Man's best friend?", "Dog."),
    ]
    pairs.extend(animals)
    
    # === ECHO PATTERNS ===
    echo_words = ["apple", "banana", "orange", "computer", "phone", "book", "table", 
                  "chair", "window", "door", "house", "car", "tree", "flower", "river",
                  "mountain", "ocean", "sky", "sun", "moon", "star", "cloud", "rain",
                  "snow", "fire", "earth", "air", "stone", "metal", "glass", "wood"]
    for word in echo_words:
        pairs.append((f"Repeat: {word}.", f"{word.capitalize()}."))
        pairs.append((f"Echo: {word}.", f"{word.capitalize()}."))
    
    # === YES/NO QUESTIONS ===
    yes_no = [
        ("Is water wet?", "Yes."),
        ("Is fire cold?", "No."),
        ("Is the sky blue?", "Yes."),
        ("Is grass green?", "Yes."),
        ("Is ice hot?", "No."),
        ("Is 2+2=4?", "Yes."),
        ("Is 2+2=5?", "No."),
        ("Is Earth flat?", "No."),
        ("Is Earth round?", "Yes."),
        ("Is Python a language?", "Yes."),
        ("Is Java a drink?", "Yes."),
        ("Is Java a language?", "Yes."),
        ("Is 10 > 5?", "Yes."),
        ("Is 3 > 7?", "No."),
        ("Is Monday a day?", "Yes."),
        ("Is 0 positive?", "No."),
        ("Is 1 odd?", "Yes."),
        ("Is 2 even?", "Yes."),
        ("Is red a color?", "Yes."),
        ("Is silence loud?", "No."),
    ]
    pairs.extend(yes_no)
    
    # === GREETINGS ===
    greetings = [
        ("Good morning!", "Good morning!"),
        ("Good afternoon!", "Good afternoon!"),
        ("Good evening!", "Good evening!"),
        ("Good night!", "Good night!"),
        ("Hello!", "Hello!"),
        ("Hi there!", "Hi!"),
        ("Hey!", "Hey!"),
        ("Greetings!", "Greetings!"),
        ("Howdy!", "Howdy!"),
        ("What's up?", "Hello!"),
    ]
    pairs.extend(greetings)
    
    # === POLITE RESPONSES ===
    polite = [
        ("Thank you.", "You're welcome."),
        ("Thanks!", "You're welcome!"),
        ("I appreciate it.", "You're welcome."),
        ("Sorry.", "No problem."),
        ("My apologies.", "No worries."),
        ("Excuse me.", "Of course."),
        ("Please help.", "Sure."),
        ("Can you help?", "Yes."),
        ("Will you help?", "Yes."),
        ("Help me.", "OK."),
    ]
    pairs.extend(polite)
    
    # === STATUS CHECKS ===
    status = [
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
    ]
    pairs.extend(status)
    
    # === COUNTING ===
    for i in range(1, 21):
        pairs.append((f"Count to {i}, last number only.", f"{i}."))
    
    # === LETTERS ===
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, letter in enumerate(alphabet):
        pairs.append((f"Letter number {i+1}?", f"{letter}."))
        pairs.append((f"Say letter {letter}.", f"{letter}."))
    
    # === COMPARISONS ===
    comparisons = [
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
    ]
    pairs.extend(comparisons)
    
    # === FILE EXTENSIONS ===
    extensions = [
        ("Python file extension?", ".py"),
        ("JavaScript extension?", ".js"),
        ("HTML extension?", ".html"),
        ("CSS extension?", ".css"),
        ("JSON extension?", ".json"),
        ("Markdown extension?", ".md"),
        ("Text file extension?", ".txt"),
        ("Image PNG extension?", ".png"),
        ("Image JPEG extension?", ".jpg"),
        ("PDF extension?", ".pdf"),
    ]
    pairs.extend(extensions)
    
    # === PROGRAMMING BASICS ===
    programming = [
        ("Print in Python?", "print()"),
        ("Comment in Python?", "#"),
        ("Comment in JavaScript?", "//"),
        ("Boolean true?", "True."),
        ("Boolean false?", "False."),
        ("Empty in Python?", "None."),
        ("Empty in JavaScript?", "null."),
        ("List in Python?", "[]"),
        ("Dict in Python?", "{}"),
        ("String quote?", "Quotes."),
    ]
    pairs.extend(programming)
    
    # === MATH SYMBOLS ===
    math_symbols = [
        ("Plus sign?", "+"),
        ("Minus sign?", "-"),
        ("Multiply sign?", "*"),
        ("Divide sign?", "/"),
        ("Equals sign?", "="),
        ("Greater than?", ">"),
        ("Less than?", "<"),
        ("Not equal?", "!="),
        ("Modulo sign?", "%"),
        ("Power sign?", "**"),
    ]
    pairs.extend(math_symbols)
    
    # === UNITS ===
    units = [
        ("Meters abbreviation?", "m"),
        ("Kilometers abbreviation?", "km"),
        ("Centimeters abbreviation?", "cm"),
        ("Kilograms abbreviation?", "kg"),
        ("Grams abbreviation?", "g"),
        ("Seconds abbreviation?", "s"),
        ("Minutes abbreviation?", "min"),
        ("Hours abbreviation?", "h"),
        ("Celsius abbreviation?", "°C"),
        ("Percent sign?", "%"),
    ]
    pairs.extend(units)
    
    # === FINISH/COMPLETE PATTERNS ===
    finish = [
        ("End response.", "Done."),
        ("Stop here.", "Stopped."),
        ("That's all.", "OK."),
        ("Finished?", "Yes."),
        ("Complete?", "Yes."),
        ("All done?", "Yes."),
        ("Nothing more.", "OK."),
        ("End.", "End."),
        ("Terminate.", "Terminated."),
        ("Close.", "Closed."),
    ]
    pairs.extend(finish)
    
    # === VARIED INSTRUCTION PATTERNS ===
    varied = [
        ("One word answer: success.", "Success."),
        ("Single word: failure.", "Failure."),
        ("Brief: hello.", "Hello."),
        ("Short: goodbye.", "Goodbye."),
        ("Quick: yes.", "Yes."),
        ("Fast: no.", "No."),
        ("Simple: OK.", "OK."),
        ("Minimal: done.", "Done."),
        ("Concise: confirmed.", "Confirmed."),
        ("Terse: denied.", "Denied."),
    ]
    pairs.extend(varied)
    
    return pairs


def create_episode(question: str, answer: str, episode_id: int) -> dict:
    """Create a single episode in the expected format."""
    return {
        "system": SYSTEM_PROMPT,
        "context": f"episode_id: format_lock_{episode_id:04d}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "language": "en"
    }


def main():
    output_dir = Path("data/sft_format_lock")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mypt_format_lock_v1.jsonl"
    
    # Generate pairs
    pairs = generate_pairs()
    
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
    
    print(f"Generated {total_pairs} Q&A pairs")
    print(f"Output: {output_file}")
    print(f"Average question length: {avg_q_len:.1f} chars")
    print(f"Average answer length: {avg_a_len:.1f} chars")
    print()
    print("Sample pairs:")
    for i in range(10):
        q, a = pairs[i]
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print()


if __name__ == "__main__":
    main()
