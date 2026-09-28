
import streamlit as st
import pickle
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Mutual Fund Risk Analyzer",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# NLP SETUP
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return words


def text_preprocessor(text):
    return ' '.join(preprocess_text(text))


# --------------------------------------------------
# LOAD SAVED MODEL AND VECTORIZER
# --------------------------------------------------

with open("intent_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# --------------------------------------------------
# CHATBOT RESPONSES
# --------------------------------------------------

responses = {
    "MUTUAL_FUND_INFO":
        "A mutual fund is an investment vehicle that pools money from many investors and invests it in assets such as stocks, bonds, or other securities.",

    "SIP_INFO":
        "SIP stands for Systematic Investment Plan. It allows an investor to invest a fixed amount at regular intervals, such as monthly, into a mutual fund.",

    "NAV_INFO":
        "NAV stands for Net Asset Value. It represents the per-unit value of a mutual fund based on the value of its assets minus liabilities.",

    "RISK_INFO":
        "Mutual funds involve different levels of risk depending on the assets held by the fund. Equity-oriented funds generally experience more market fluctuation than many debt-oriented funds.",

    "EQUITY_FUND_INFO":
        "Equity mutual funds primarily invest in stocks. Their value can fluctuate significantly with market conditions.",

    "DEBT_FUND_INFO":
        "Debt mutual funds primarily invest in fixed-income securities such as bonds and money-market instruments.",

    "HYBRID_FUND_INFO":
        "Hybrid mutual funds invest in a combination of asset classes, such as equity and debt.",

    "BEGINNER_GUIDANCE":
        "If you are new to mutual funds, first understand concepts such as risk, returns, diversification, expense ratio, SIP, and investment horizon.",

    "RISK_PROFILE":
        "Risk profiling helps understand an investor's tolerance for investment fluctuations and potential losses.",

    "ELIGIBILITY_INFO":
        "Eligibility to invest in mutual funds can depend on factors such as age, legal status, and the type of account used. For minors or school students, investment may require a parent or legal guardian and specific account arrangements. Check the current applicable rules before investing.",

    "RETURNS_INFO":
        "Mutual fund returns depend on the assets held by the fund and market conditions. Returns are not guaranteed, and past performance does not guarantee future results.",

    "EXPENSE_RATIO_INFO":
        "The expense ratio is the annual fee charged by a mutual fund for managing and operating the fund. It is expressed as a percentage of the fund's assets and can affect the investor's net returns.",

    "DIVERSIFICATION_INFO":
        "Diversification means spreading investments across different assets or securities rather than depending on a single investment. It can help reduce the impact of poor performance from one particular investment, although it does not eliminate risk.",

    "INVESTMENT_HORIZON":
        "Investment horizon means the length of time an investor expects to remain invested. The appropriate horizon depends on the investment goal, financial situation, and risk tolerance.",

    "TAX_INFO":
        "Mutual fund investments and gains may be subject to taxation depending on factors such as the type of fund, holding period, and applicable tax rules. Tax rules can change, so investors should check the current rules or consult a qualified tax professional."
}


# --------------------------------------------------
# CHATBOT FUNCTION
# --------------------------------------------------

def predict_intent(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    return prediction


def chatbot_response(user_text):
    intent = predict_intent(user_text)

    return responses.get(
        intent,
        "Sorry, I don't understand your question yet."
    )


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("📊 Mutual Fund Risk Analyzer")
st.subheader("🤖 Investor Guidance Chatbot")

st.write(
    "Ask questions about mutual funds, SIPs, NAV, risk, returns, "
    "diversification, taxes, and other investment concepts."
)

st.divider()

user_question = st.text_input(
    "💬 Ask your question:"
)

if user_question:

    intent = predict_intent(user_question)
    answer = chatbot_response(user_question)

    st.write("### 🤖 Chatbot Response")
    st.success(answer)

    st.caption(f"Detected topic: {intent}")


st.divider()

st.warning(
    "⚠️ Educational use only. This application does not provide "
    "personalized financial advice or recommendations to buy or sell investments."
)
