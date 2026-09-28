import os
from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient
from openai import OpenAI
#from anthropic import Anthropic

load_dotenv()

#--- Jev Testing ---

jev = TypeSafeClient(api_key=os.environ["TYPESAFE_API"])
jev_response = jev.system_one(
    state='ping', 
    questions={
        "ok": Choice (
            instructions="Is this a working connection?",
            criteria={"yes": "Connections Works", "no": "Connetion Failed"},
        ),
    },
)

print("Jev:", jev_response.answers["ok"].choice, jev_response.answers["ok"].confidence)

#--- Deepseek Testing ---

deepseek = OpenAI(
    api_key=os.environ["DEEPSEEK_API"],
    base_url="https://api.deepseek.com"
)
fast_response = deepseek.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Say 'ok' if you're working"}],
)

print("Deepseek:", fast_response.choices[0].message.content)