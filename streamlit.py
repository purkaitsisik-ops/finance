# streamlit_app.py
import streamlit as st
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import numpy as np
import os
import json
import re

# ===== CONFIG =====
st.set_page_config(page_title="Derivatives Monte Carlo Pricer", page_icon="💹", layout="centered")

# User API Key
api_key = st.text_input("Enter your Google API Key", type="password")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Market parameters (editable in sidebar)
st.sidebar.header("Market Parameters")
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
n_paths = st.sidebar.number_input("Monte Carlo Paths", value=100000, step=10000)
steps = st.sidebar.number_input("Steps per year", value=252)

# ===== LangChain LLM =====
if api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    prompt = PromptTemplate(
    input_variables=["statement"],
    template=(
        "You are a derivatives payoff interpreter.\n"
        "Input: {statement}\n"
        "Output only in JSON with keys:\n"
        "- formula: Python expression for payoff in terms of:\n"
        "  * S_path (NumPy array of daily prices)\n"
        "  Example: max(np.mean(S_path) - 100, 0)\n"
        "- T: time to maturity in years (float)\n"
        "Do not add explanations."
    )
    )


    def get_payoff_from_llm(statement):
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        response = chain.invoke({"statement": statement})

        return response  # JSON string




    def mc_price(formula_str, T):
    # Convert formula to a Python lambda
        dt = 1 / steps
        n_steps = int(T * steps)

        # Preallocate paths
        S_paths = np.zeros((n_paths, n_steps + 1))
        S_paths[:, 0] = S0

        # Simulate Geometric Brownian Motion
        for t in range(1, n_steps + 1):
            z = np.random.randn(n_paths)
            S_paths[:, t] = S_paths[:, t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )

        # Turn formula into a lambda that takes the full path
        # The LLM will produce expressions like: max(np.mean(S_path) - 100, 0)
        payoff_fn = eval(f"lambda S_path: {formula_str}", {"np": np, "max": max, "min": min})

        # Apply payoff to each path
        payoffs = np.array([payoff_fn(path) for path in S_paths])

        # Discount back
        return np.exp(-r * T) * np.mean(payoffs)
    st.title("💹 Derivatives Monte Carlo Pricer")
    st.markdown("Enter a **natural language** description of a payoff, and the app will interpret it, "
                "convert it into a formula, and price it using Monte Carlo simulation.")

    statement = st.text_area("Enter payoff description:", 
                              "The product pays excess of final price square minus initial price(S0) or 0 if it is negative with time period 19 years")

    if st.button("Price Option"):
        with st.spinner("Interpreting payoff..."):
            data = get_payoff_from_llm(statement)

        # Try parsing JSON
        

        formula = data["formula"].replace("^", "**")
        T = float(data["T"])

        st.write("**Interpreted Payoff Formula**")
        st.write("**Time to Maturity:**", T, "years")

        with st.spinner("Running Monte Carlo simulation..."):
            try:
                price = mc_price(formula, T)
                st.success(f"Monte Carlo Price: {price:.4f}")
            except Exception as e:
                st.error(f"Error evaluating formula: {e}")

else:
    st.warning("Please enter your Google API key to use the app.")
