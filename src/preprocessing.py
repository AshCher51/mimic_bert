import string
import re
import pandas as pd

notes = pd.read_csv("../data/notes.csv")
labels = pd.read_csv("../data/labels.csv")

# print(notes.head())
# print(labels.head())

# print(list(notes.columns))
# print("\n")
# print(list(labels.columns))

notes.drop(
    [
        "Unnamed: 0",
        "ROW_ID",
        "CHARTDATE",
        "CHARTTIME",
        "STORETIME",
        "CATEGORY",
        "DESCRIPTION",
        "CGID",
        "ISERROR",
        "HADM_ID",
    ],
    axis=1,
    inplace=True,
)
labels.drop(["ROW_ID", "BATCH.ID", "OPERATOR", "HADM_ID"], axis=1, inplace=True)


cleaned_text_list = []
for note in notes["TEXT"]:
    cleaned_text = note.translate(str.maketrans("", "", string.punctuation))
    cleaned_text = re.sub("\n", "", cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text_list.append(re.sub(r"[0-9]+", "", note))

notes["TEXT"] = pd.Series(cleaned_text_list)

result = pd.merge(notes, labels, how="inner", on=["SUBJECT_ID"])
result.columns = [
    "SUBJECT_ID",
    "TEXT",
    "CANCER",
    "HEART DISEASE",
    "LUNG DISEASE",
    "ALCOHOL ABUSE",
    "CHRONIC_NEURO_DYSTROPHIES",
    "CHRONIC PAIN",
    "DEMENTIA",
    "DEPRESSION",
    "DEVELOPMENT RETARDATION",
    "NON ADHERENCE",
    "NONE",
    "OBESITY",
    "SUBSTANCE ABUSE",
    "PSYCHIATRIC DISORDERS",
    "UNSURE",
]

print(result.head())
result.to_csv("../data/temp.csv")
result.to_csv("temp.csv")
