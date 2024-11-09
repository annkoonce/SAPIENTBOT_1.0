class LanguageProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

    def _setup_patterns(self):
        greetings = [{"LOWER": "hello"}, {"LOWER": "hi"}, {"LOWER": "hey"}]
        farewells = [{"LOWER": "bye"}, {"LOWER": "goodbye"}, {"LOWER": "see"}, {"LOWER": "you"}]
        self.matcher.add("GREETING", [greetings])
        self.matcher.add("FAREWELL", [farewells])

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return "positive"
        elif analysis.sentiment.polarity < 0:
            return "negative"
        else:
            return "neutral"