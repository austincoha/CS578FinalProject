import os
from email import policy
from email.parser import BytesParser

def extract_clean_plain_text(msg):
    """Extract top-level plain text from a multipart email."""
    for part in msg.iter_parts():
        if part.get_content_type() == 'text/plain':
            content = part.get_content().strip()
            return extract_top_level_text(content)
    return None

def extract_top_level_text(body):
    """Remove quoted reply/forward chains to isolate the original message."""
    reply_separators = [
        "-----Original Message-----",
        "\nFrom:",
        "\nSent:",
        "\nTo:",
        "\nSubject:"
    ]
    for sep in reply_separators:
        if sep in body:
            return body.split(sep, 1)[0].strip()
    return body.strip()

def process_single_email_file(filepath):
    """Parse a single email file and return cleaned plain text (or None)."""
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_data)
    except Exception:
        return None

    if not msg.is_multipart():
        return None  # skip if not multipart

    return extract_clean_plain_text(msg)

def process_all_email_files(folder_path, output_path="combined_cleaned_emails.txt"):
    """Process all no-extension email files and combine cleaned texts."""
    all_cleaned = []
    for filename in os.listdir(folder_path):
        if '.' not in filename:  # files with no extension
            filepath = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            cleaned = process_single_email_file(filepath)
            if cleaned:
                all_cleaned.append(cleaned)

    with open(os.path.join(folder_path, output_path), 'w', encoding='utf-8') as f:
        for i, email in enumerate(all_cleaned, 1):
            f.write(f"--- Email #{i} ---\n")
            f.write(email + "\n\n")

    print(f"\n✅ Saved {len(all_cleaned)} cleaned email(s) to: {output_path}")

# Run it
process_all_email_files("data/datasets")