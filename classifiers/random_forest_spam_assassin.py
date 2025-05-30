from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

nlp = spacy.load("en_core_web_sm")

PREPROCESSING_METHOD = 1


def load_and_prepare_data():
    data = load_dataset("talby/spamassassin", "text")
    df = pd.DataFrame(data['train'])
    # print("Data Shape:", df.shape)
    # print("Label Distribution:\n", df["label"].value_counts())
    # print("Group Distribution:\n", df["group"].value_counts())
    # print(df.head())
    return df


def split_data(df):
    hard_ham_df = df[df["group"] == "hard_ham"]
    standard_df = df[df["group"] != "hard_ham"]

    x = standard_df["text"]
    y = standard_df["label"]

    x_train, x_test_part, y_train, y_test_part = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=17
    )

    x_test = pd.concat([x_test_part, hard_ham_df["text"]])
    y_test = pd.concat([y_test_part, hard_ham_df["label"]])

    # Align indices properly
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return x_train, y_train, x_test, y_test, df

def clean_with_spacy(text):
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and not token.like_url and not token.like_email and not token.like_num
    ]
    return " ".join(tokens)

def preprocess_data(x_train, x_test):
    vectorizer = TfidfVectorizer(
        preprocessor=clean_with_spacy,
        stop_words='english'
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return x_train_vec, x_test_vec, vectorizer

def train_model(x_train_vec, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=71)
    model.fit(x_train_vec, y_train)
    return model


def evaluate_model(model, x_test_vec, y_test, x_test, df):
    y_pred = model.predict(x_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    results_df = pd.DataFrame({
        "row_index": range(len(x_test)),
        "text": x_test,
        "true_label": y_test,
        "predicted_label": y_pred
    })

    # Merge group info by matching on text content
    results_df = results_df.merge(df[["text", "group"]], on="text", how="left")

    return results_df


def save_and_report(results_df):
    results_df.to_csv("spam_predictions.csv", index=False)
    print("\nSaved all predictions to 'spam_predictions.csv'")

    # Create a new column for correct predictions
    results_df["correct"] = results_df["true_label"] == results_df["predicted_label"]

    # Compute group accuracy without using apply
    group_accuracy = (
        results_df
        .groupby("group")["correct"]
        .mean()
        .reset_index(name="accuracy")
    )

    print("\nPer-Group Accuracy:")
    print(group_accuracy)

    print("\nCorrect Predictions:")
    print(results_df[results_df["correct"]]
          [["row_index", "group", "true_label", "predicted_label", "text"]]
          .sample(5, random_state=42))

    print("\nIncorrect Predictions:")
    print(results_df[~results_df["correct"]]
          [["row_index", "group", "true_label", "predicted_label", "text"]]
          .sample(5, random_state=42))


def main():
    df = load_and_prepare_data().head(10) # restore later
    x_train, y_train, x_test, y_test, df = split_data(df)
    print(x_train[0])
    print(clean_with_spacy(x_train[0]))
    x_train_vec, x_test_vec, _ = preprocess_data(x_train, x_test)

    # model = train_model(x_train_vec, y_train)
    # results_df = evaluate_model(model, x_test_vec, y_test, x_test, df)
    # save_and_report(results_df)


if __name__ == "__main__":
    main()
