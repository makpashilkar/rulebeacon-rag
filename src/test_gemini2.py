from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
result=model.invoke('eply with exactly: Gemini connection works.')
print(result.content[0]['text'])