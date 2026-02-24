# ai_module.py
import google.generativeai as genai

# ====== Replace with your actual Gemini API key ======
API_KEY = "AIzaSyAQKrnVJOjv0EpRlOAIifIsflM-SnTd2fk"

# Configure Gemini with your API key
genai.configure(api_key=API_KEY)

def ask_gemini(question, context=""):
    """
    Ask Gemini AI for market insights.
    :param question: User question string
    :param context: Optional chart/signal context
    :return: AI response text
    """
    if not API_KEY:
        raise ValueError("Missing API_KEY in ai_module.py. Please set your Gemini API key.")

    # Initialize the Gemini model (you can change to 'gemini-1.5-pro' or 'gemini-1.5-flash')
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Construct the full prompt
    prompt = f"""
    You are a professional AI analyst specialized in real-time technical analysis and crypto market insights.

    Context: {context}

    Question: {question}
    """

    # Generate the response
    response = model.generate_content(prompt)

    # Return the text output
    return response.text
