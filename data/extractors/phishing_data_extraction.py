import os
from email import policy
from email.parser import BytesParser

def extract_clean_plain_text(msg):
    for part in msg.iter_parts():
        if part.get_content_type() == 'text/plain':
            return part.get_content().strip()
    return None

def process_email_file(filepath, limit=100000):
    with open(filepath, 'rb') as f:
        raw_data = f.read()

    # Prepare output file path
    directory = os.path.dirname(filepath)
    output_path = os.path.join(directory, "phishing-2024_plaintext_output.txt")

    # Split on the start of each email (but keep the split string)
    raw_emails = raw_data.split(b'\nFrom jose@monkey.org')
    count = 0

    with open(output_path, 'w', encoding='utf-8') as out_file:
        for i, raw_email in enumerate(raw_emails):
            if not raw_email.strip():
                continue  # skip any empty chunks

            # Add back the first line (only if it's not the first chunk)
            if i > 0:
                raw_email = b'From jose@monkey.org' + raw_email

            try:
                msg = BytesParser(policy=policy.default).parsebytes(raw_email)
            except Exception:
                continue  # skip parse errors

            if not msg.is_multipart():
                continue  # skip non-multipart messages

            text = extract_clean_plain_text(msg)
            if text:
                count += 1
                out_file.write(f"\n--- Email #{count} ---\n{text}\n")

            if count >= limit:
                break

    print(f"Wrote {count} email(s) to: {output_path}")

# Run the parser
process_email_file("data/datasets/phishing-2024.txt")
