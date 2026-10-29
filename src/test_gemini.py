from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.8-flash', temperature=0)
result=model.invoke('eply with exactly: Gemini connection works.')
print(result.content)