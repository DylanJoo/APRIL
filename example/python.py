from reranking.wrapper import ModularReranker
import ir_measures
from ir_measures import RR

# Example data
run = {
    "q1":  {"d2": 1.00, "d1*": 0.62, "d13*": 0.18, "d31*": 0.12},
    "q2":  {"d4": 0.91, "d3*": 0.77, "d28": 0.21, "d16": 0.15},
    "q3":  {"d6": 0.88, "d5*": 0.71, "d21*": 0.24, "d22": 0.19},
    "q4":  {"d8": 0.82, "d7*": 0.79, "d20": 0.33, "d19*": 0.17},
    "q5":  {"d10": 0.95, "d9*": 0.74, "d34": 0.22, "d33*": 0.11},
    "q6":  {"d12": 0.83, "d11*": 0.80, "d22": 0.28, "d21*": 0.10},
    "q7":  {"d14": 0.97, "d13*": 0.69, "d1*": 0.20, "d32": 0.18},
    "q8":  {"d16": 0.86, "d15*": 0.77, "d28": 0.25, "d4": 0.12},
    "q9":  {"d18": 0.90, "d17*": 0.66, "d26": 0.14, "d30": 0.09},
    "q10": {"d20": 0.92, "d19*": 0.81, "d8": 0.27, "d27*": 0.08},
    "q11": {"d24": 0.93, "d23*": 0.76, "d34": 0.16, "d29*": 0.07},
    "q12": {"d26": 0.87, "d25*": 0.79, "d36": 0.22, "d4": 0.10},
    "q13": {"d28": 0.89, "d27*": 0.75, "d16": 0.20, "d25*": 0.05},
    "q14": {"d30": 0.91, "d29*": 0.78, "d24": 0.18, "d33*": 0.06},
    "q15": {"d32": 0.96, "d31*": 0.73, "d14": 0.19, "d1*": 0.11},
    "q16": {"d34": 0.89, "d33*": 0.82, "d24": 0.21, "d10": 0.09},
    "q17": {"d36": 0.88, "d35*": 0.80, "d26": 0.23, "d30": 0.07}
}

queries = {
    "q1":  "What is the capital of France?",
    "q2":  "Whi wrote 'Pride and Prejudice'?",                  # typo + quotes
    "q3":  "At sea level, what's water's boiling point (°C)?",  # paraphrase + unit
    "q4":  "Which planet is nicknamed the Red Planet?",         # alias
    "q5":  "d/dx sin(x) equals what?",                          # math notation
    "q6":  "What's the chemical symbol for gold?",              # symbol vs. name
    "q7":  "Capital of Japan?",                                  # short form
    "q8":  "¿Quién escribió 'Cien años de soledad'?",           # cross-lingual ES
    "q9":  "What percent voted Leave in the 2016 Brexit referendum?", # numeric fact
    "q10": "Which mountain is the tallest above sea level?",    # comparative
    "q11": "Binary representation of 13?",                      # exact numeric
    "q12": "Which company did Steve Jobs co-found?",            # entity linking
    "q13": "Who painted the Mona Lisa?",                        # famous work
    "q14": "What's π approximately?",                           # constant
    "q15": "What's Canada's capital?",                          # country/city confuser
    "q16": "√144 equals?",                                      # symbol variant
    "q17": "Which language did Guido van Rossum create?"        # creator mapping
}

corpus = {
    "d1*":  "Paris is the capital of France.",
    "d2":   "London is the capital of the UK.",
    "d3*":  "'Pride and Prejudice' was written by Jane Austen.",
    "d4":   "'1984' was written by George Orwell.",
    "d5*":  "The boiling point of water at sea level is 100°C.",
    "d6":   "Water freezes at 0°C.",
    "d7*":  "The Red Planet is Mars.",
    "d8":   "Jupiter is the largest planet.",
    "d9*":  "The derivative of sin(x) is cos(x).",
    "d10":  "The integral of sin(x) is -cos(x) + C.",
    "d11*": "The chemical symbol for gold is Au.",
    "d12":  "Ag is the chemical symbol for silver.",
    "d13*": "The capital of Japan is Tokyo.",
    "d14":  "Kyoto is a historic city in Japan.",
    "d15*": "El autor de 'Cien años de soledad' es Gabriel García Márquez.",
    "d16":  "'Don Quixote' was written by Miguel de Cervantes.",
    "d17*": "In 2016, the Brexit referendum resulted in 51.9% voting to Leave.",
    "d18":  "The European Union currently has 27 member states.",
    "d19*": "The tallest mountain on Earth above sea level is Mount Everest.",
    "d20":  "K2 is the second-highest mountain on Earth.",
    "d21*": "H2O is the molecular formula of water.",
    "d22":  "CO2 is the molecular formula of carbon dioxide.",
    "d23*": "In binary, 13 is 1101.",
    "d24":  "In binary, 12 is 1100.",
    "d25*": "The company co-founded by Steve Jobs is Apple.",
    "d26":  "Microsoft was founded by Bill Gates and Paul Allen.",
    "d27*": "The painter of the Mona Lisa was Leonardo da Vinci.",
    "d28":  "Vincent van Gogh painted 'The Starry Night'.",
    "d29*": "π (pi) is approximately 3.14159.",
    "d30":  "e (Euler's number) is approximately 2.71828.",
    "d31*": "The capital of Canada is Ottawa.",
    "d32":  "Toronto is the largest city in Canada.",
    "d33*": "The square root of 144 is 12.",
    "d34":  "The square of 12 is 144.",
    "d35*": "The programming language created by Guido van Rossum is Python.",
    "d36":  "Java was created by James Gosling."
}

qrel = {
    "q1":  {"d1*": 1},
    "q2":  {"d3*": 1},
    "q3":  {"d5*": 1},
    "q4":  {"d7*": 1},
    "q5":  {"d9*": 1},
    "q6":  {"d11*": 1},
    "q7":  {"d13*": 1},
    "q8":  {"d15*": 1},
    "q9":  {"d17*": 1},
    "q10": {"d19*": 1},
    "q11": {"d23*": 1},
    "q12": {"d25*": 1},
    "q13": {"d27*": 1},
    "q14": {"d29*": 1},
    "q15": {"d31*": 1},
    "q16": {"d33*": 1},
    "q17": {"d35*": 1}
}

# Initialize the reranker
reranker = ModularReranker.from_prebuilt('rankgpt', 'Qwen/Qwen2.5-7B-Instruct')
reranked_run = reranker.rerank(run=run, queries=queries, corpus=corpus)

# Evaluation
print(ir_measures.calc_aggregate([RR@5], qrel, run))
print(ir_measures.calc_aggregate([RR@5], qrel, reranked_run))
# {RR@5: 0.5}
# {RR@5: 1.0}
