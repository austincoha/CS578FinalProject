import fs from 'fs/promises';
import OpenAI from 'openai';

import dotenv from 'dotenv';
dotenv.config();

// Initialize the client (will pull API key from process.env.OPENAI_API_KEY)
const openai = new OpenAI();

async function classifyEmail(emailText) {
  const prompt = [
    { role: 'system', content: 'You are an email security analyst.' },
    { role: 'user', content:
      `Determine whether the following email is ‘Phishing’ or ‘Legitimate’, then give a one-sentence rationale.\n\n` +
      `---\n${emailText}\n---\n` +
      `Answer format:\n1. Label: <Phishing|Legitimate>\n2. Rationale: <…>`
    }
  ];

  const resp = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: prompt,
    temperature: 0.0
  });

  return resp.choices[0].message.content.trim();
}

async function main() {
  try {
    // Read and split your emails file
    const data = await fs.readFile('emails.txt', 'utf-8');
    const emails = data.split('NEXTEMAIL'); // adjust delimiter if needed

    for (let i = 0; i < emails.length; i++) {
      const email = emails[i].trim();
      if (!email) continue;

      const result = await classifyEmail(email);
      console.log(`Email ${i+1} → ${result}\n`);
    }
  } catch (err) {
    console.error('Error:', err);
  }
}

main();
