from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# https://huggingface.co/datasets/talby/spamassassin
spamassassin_data = load_dataset("talby/spamassassin", "text")
df = pd.DataFrame(spamassassin_data['train'])
df = df.reset_index(drop=False)

# Preview Data
print("Data Shape:", df.shape)
print("Label Distribution:\n", df["label"].value_counts())
print("Group Distribution:\n", df["group"].value_counts())
print(df.head())

hard_ham_df = df[df["group"] == "hard_ham"]
standard_df = df[df["group"] != "hard_ham"]

# x = standard_df["text"]
# y = standard_df["label"]
# indices = standard_df["index"]
x = df["text"]
y = df["label"]
indices = df["index"]

x_train, x_test_part, y_train, y_test_part, idx_train, idx_test_part = train_test_split(
    x, y, indices, test_size=0.2, stratify=y, random_state=17
)

x_test = pd.concat([x_test_part, hard_ham_df["text"]], ignore_index=True)
y_test = pd.concat([y_test_part, hard_ham_df["label"]], ignore_index=True)
idx_test = pd.concat([idx_test_part, hard_ham_df["index"]], ignore_index=True)

vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = RandomForestClassifier(n_estimators=100, random_state=71)
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

results_df = pd.DataFrame({
    "index": idx_test,
    "text": x_test,
    "true_label": y_test,
    "predicted_label": y_pred
})
results_df = results_df.merge(df[["index", "group"]], on="index", how="left")

results_df.to_csv("spam_predictions.csv", index=False)
print("\nSaved all predictions to 'spam_predictions.csv'")

group_accuracy = (
    results_df
    .groupby("group")
    .apply(lambda g: (g["true_label"] == g["predicted_label"]).mean())
    .reset_index(name="accuracy")
)

# Print per-group accuracy
print("\nPer-Group Accuracy:")
print(group_accuracy)

# Show correct predictions
correct_preds = results_df[results_df["true_label"] == results_df["predicted_label"]]
print("\nCorrect Predictions:")
print(correct_preds[["group", "true_label", "predicted_label", "text"]].sample(5, random_state=42))

# Show incorrect predictions
incorrect_preds = results_df[results_df["true_label"] != results_df["predicted_label"]]
print("\nIncorrect Predictions:")
print(incorrect_preds[["group", "true_label", "predicted_label", "text"]].sample(5, random_state=42))
