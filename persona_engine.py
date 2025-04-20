import os
import openai
import json
import google.generativeai as genai
import requests
from openai import OpenAI
PROMPT_TEMPLATE = """
You are a Persona Engine that takes structured input from a customer interaction context and outputs the following fields:
1. Role: A paragraph describing the AI agent's persona, including:
    - A name that fits the situation
    - Communication style inspired by a known personality or archetype (e.g., calm advisor, energetic peer, etc.)
    - Tone appropriate to the customer’s persona (e.g., mildly irritated, impatient, young, senior, etc.)
    - A brief about the company (Lenskart, an online eyewear company in India)
    - Domain knowledge they are expected to provide (e.g., order status, warranties, returns, etc.)

2. Instructions: A list of tone/behavioral guidelines for how the agent should respond in this conversation. Think of this as a style guide for language, empathy, assertiveness, and formality.

3. Objective: A list of 2-5 clear objectives for this interaction. These should include what the AI agent should achieve or communicate in this specific exchange. Be specific, e.g., "Reassure the customer about resolution within 24 hours", or "Politely ask for order ID".

Based on the following input, return only a JSON object in the following schema:
{{
  "Role": "Your role description here",
  "Instructions": ["Sentence 1", "Sentence 2",...],
  "Objective": ["Goal 1", "Goal 2",...]
}}

### Input:
Interaction Number: {interaction_number}
Steps Remaining: {steps_remaining}
Timeframe Left: {timeframe}
Interaction Reason: {interaction_reason}
Customer Age: {customer_age}
Previous Orders: {previous_orders}
Customer Persona: {customer_persona}

Return only the JSON. No explanation or intro.
"""


class PersonaEngine:
    def __init__(self, provider='openai'):
        self.provider = provider
        if provider == 'openai':
            openai_ap_key=None
            with open('creds.json', 'r') as file:
                data = json.load(file)
                openai_ap_key= data['OPENAI_API_KEY']
            openai.api_key = openai_ap_key
        elif provider == 'gemini':
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        elif provider == 'sambanova':
            self.sambanova_key = os.environ.get("SAMBANOVA_API_KEY")

    def get_prompt(self, inputs):
        print(PROMPT_TEMPLATE.format(**inputs))
        return PROMPT_TEMPLATE.format(**inputs)

    def generate(self, inputs):
        prompt = self.get_prompt(inputs)
        if self.provider == 'openai':
            return self._from_openai(prompt)
        elif self.provider == 'gemini':
            return self._from_gemini(prompt)
        elif self.provider == 'sambanova':
            return self._from_sambanova(prompt)
        else:
            raise ValueError("Unsupported provider")

    def _from_openai(self, prompt):
        openai_ap_key=None
        with open('creds.json', 'r') as file:
            data = json.load(file)
            openai_ap_key= data['OPENAI_API_KEY']

        client = OpenAI(
            # This is the default and can be omitted
            api_key=openai_ap_key,
        )

        print(prompt)
        response = client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant",
            input=prompt,
        )
        text = response.output_text
        start_index = text.find('{')
        end_index = text.rfind('}') + 1 # +1 to include the closing curly brace
        json_string = text[start_index:end_index]
        data = json.loads(json_string)
        print("Text:", text)
        print("Data:", data)
        return data

    def _from_gemini(self, prompt):
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return json.loads(response.text.strip())

    def _from_sambanova(self, prompt):
        url = "https://api.sambanova.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.sambanova_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=payload)
        return json.loads(response.json()['text'].strip())
