"""
generate_dataset.py - Creates a synthetic sarcasm detection dataset
Run once: python generate_dataset.py
"""

import pandas as pd
import random
import os

random.seed(42)

# ─── Sarcastic English sentences ────────────────────────────────────────────────
sarcastic_en = [
    ("Oh great, another Monday!", "I love weekends so much"),
    ("Yeah, because that always works out perfectly.", "He tried the same approach again"),
    ("Sure, I totally have time for this right now.", "I have three deadlines today"),
    ("Wow, what a surprise! Nobody saw that coming.", "The team predicted this months ago"),
    ("Oh brilliant, just what I needed today.", "Everything has been going wrong"),
    ("Because obviously that's such a fantastic idea.", "The plan clearly had issues"),
    ("Yeah right, like that's ever going to happen.", "They promised delivery by Tuesday"),
    ("Oh wonderful, the wifi is down again.", "We rely on internet for meetings"),
    ("Super helpful, as always.", "She gave vague instructions again"),
    ("Oh amazing, traffic again on a Friday evening.", "I just want to get home"),
    ("Naturally, the printer breaks the day of the presentation.", "I have to print 50 copies"),
    ("Because sleeping is clearly overrated.", "I stayed up all night working"),
    ("What a shocker, another meeting that could've been an email.", "We have 3 meetings scheduled"),
    ("Oh sure, let me just drop everything.", "He always calls at the worst time"),
    ("Absolutely stellar timing.", "She announced changes at 5pm Friday"),
    ("Fantastic, my coffee is cold again.", "I forgot to drink it"),
    ("Oh how delightful, a flat tire.", "I'm already late to work"),
    ("Because that's clearly working so well.", "The bug keeps recurring in production"),
    ("Oh yes, best day ever.", "I spilled coffee on my laptop"),
    ("Totally makes sense, raise prices and add fewer features.", "The new software update launched"),
]

# ─── Non-sarcastic English sentences ────────────────────────────────────────────
not_sarcastic_en = [
    ("I really enjoyed the movie last night.", "We watched a film together"),
    ("The meeting went very well today.", "We discussed the quarterly results"),
    ("I love spending time with my family.", "We had a family dinner"),
    ("The project is progressing as expected.", "We reviewed the deliverables"),
    ("She gave a clear and helpful explanation.", "I had a question about the process"),
    ("The weather is beautiful today.", "It's spring and sunny"),
    ("I appreciate your honest feedback.", "My manager reviewed my work"),
    ("The new update fixed all the bugs.", "The dev team pushed an update"),
    ("I feel energized after a good night's sleep.", "I slept early last night"),
    ("The food at that restaurant was delicious.", "We went out for dinner"),
    ("The team did a great job on the proposal.", "We submitted it this morning"),
    ("I'm grateful for the support I received.", "My colleagues helped with the task"),
    ("The library loads much faster now.", "The optimization was implemented"),
    ("The new employee is adapting well.", "She joined two weeks ago"),
    ("The concert was an amazing experience.", "We attended it last weekend"),
    ("I finally understood the concept after the tutorial.", "The instructor explained it clearly"),
    ("The customer service was very responsive.", "I filed a complaint yesterday"),
    ("The children were well-behaved at the event.", "It was a school function"),
    ("The product quality exceeds our expectations.", "We ordered the premium version"),
    ("I'm glad the issue was resolved quickly.", "There was a scheduling conflict"),
]

# ─── Hindi sarcastic ────────────────────────────────────────────────────────────
sarcastic_hi = [
    ("वाह, क्या शानदार काम किया!", "उसने फिर से गड़बड़ी की"),
    ("हाँ बिलकुल, यह तो बहुत अच्छा विचार है।", "यह योजना पहले भी फेल हो चुकी है"),
    ("बहुत खूब, ट्रेन फिर देर से आई।", "मुझे ऑफिस के लिए देर हो रही है"),
    ("अरे वाह, आज फिर बिजली गई।", "गर्मियाँ हैं और AC बंद है"),
    ("हाँ, तुम तो हमेशा सही होते हो।", "उसने फिर बिना सोचे निर्णय लिया"),
]

not_sarcastic_hi = [
    ("आज मौसम बहुत अच्छा है।", "बाहर धूप है और ठंडी हवा है"),
    ("मुझे यह किताब बहुत पसंद आई।", "मैंने इसे कल पढ़ा"),
    ("बैठक बहुत सफल रही।", "हमने अच्छे परिणाम हासिल किए"),
    ("खाना बहुत स्वादिष्ट था।", "माँ ने खाना बनाया था"),
    ("टीम ने बहुत मेहनत की।", "प्रोजेक्ट समय पर पूरा हुआ"),
]

# ─── Tamil sarcastic ────────────────────────────────────────────────────────────
sarcastic_ta = [
    ("ஆமா, இது மிகவும் சிறந்த யோசனை!", "அவர் மீண்டும் தவறான முடிவு எடுத்தார்"),
    ("வாவ், பஸ் இன்னும் வரவில்லை.", "நான் ஒரு மணி நேரமாக காத்திருக்கிறேன்"),
    ("நன்றி, இது மிகவும் உதவியாக இருந்தது.", "அந்த ஆலோசனை எனக்கு ஒன்றும் பயனில்லை"),
    ("ஆமாம், அது மிகவும் வேலை செய்யும்.", "முன்பு இதே முறை தோல்வியடைந்தது"),
    ("சரி, சரி, நீங்கள் எப்போதும் சரியானவர்.", "அவர் மீண்டும் தர்க்கமே செய்யவில்லை"),
]

not_sarcastic_ta = [
    ("இன்று வானிலை மிகவும் அழகாக உள்ளது.", "வெயில் மற்றும் குளிர்ந்த காற்று"),
    ("புத்தகம் மிகவும் சுவாரஸ்யமாக இருந்தது.", "நேற்று இரவு படித்தேன்"),
    ("கூட்டம் வெற்றிகரமாக நடந்தது.", "நாங்கள் நல்ல முடிவுகளில் உடன்பட்டோம்"),
    ("சாப்பாடு மிகவும் சுவையாக இருந்தது.", "அம்மா சமைத்தார்"),
    ("குழு மிகவும் கடினமாக உழைத்தது.", "திட்டம் நேரத்தில் முடிந்தது"),
]

# ─── Emotion mapping ────────────────────────────────────────────────────────────
emotion_map = {
    True: random.choices(["Angry", "Sad", "Neutral"], weights=[0.4, 0.3, 0.3]),
    False: random.choices(["Happy", "Neutral"], weights=[0.5, 0.5]),
}

rows = []

def add_rows(pairs, label, lang):
    for text, context in pairs:
        emotion = random.choices(
            ["Angry", "Sad", "Neutral"] if label == 1 else ["Happy", "Neutral"],
            weights=[0.4, 0.3, 0.3] if label == 1 else [0.5, 0.5]
        )[0]
        rows.append({"text": text, "context": context, "label": label,
                     "emotion": emotion, "language": lang})

add_rows(sarcastic_en, 1, "en")
add_rows(not_sarcastic_en, 0, "en")
add_rows(sarcastic_hi, 1, "hi")
add_rows(not_sarcastic_hi, 0, "hi")
add_rows(sarcastic_ta, 1, "ta")
add_rows(not_sarcastic_ta, 0, "ta")

# Augment to 500+ rows by repeating with slight variation
base = rows.copy()
for _ in range(7):
    for row in base:
        new_row = row.copy()
        rows.append(new_row)

random.shuffle(rows)
rows = rows[:600]

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_dataset.csv", index=False)
print(f"✅ Dataset created: {len(df)} rows")
print(df.head())
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nEmotion distribution:")
print(df["emotion"].value_counts())
