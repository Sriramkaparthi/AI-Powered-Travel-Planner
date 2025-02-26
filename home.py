import streamlit as st
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

# Initialize Output Parser-LOGIC3
parser = JsonOutputParser()

# Define Prompt Template-LOGIC1
chat_template = ChatPromptTemplate(
    messages=[
        (
            "system",
            """You are a helpful AI assistant that provides approximate transportation costs from a source to a destination in Indian rupees.
            Ensure the output is structured with keys: bus, car, train, aeroplane, bike, and ship.
            If a transport mode is not available, set its value to 'Not Available'.
            For each mode, include all possible subcategories (e.g., Bus: AC Sleeper, Non-AC Sleeper, AC Push Back, Non-AC Push Back; Car: Sedan & SUV with base fare and toll charges; Train: Sleeper, 3rd AC, 2nd AC, 1st AC; Aeroplane: Economy, Business; Bike: Fuel Cost, Wear & Tear).
            You generate the output while following the below mentioned format.
            Output format instructions:{output_format_instructions}"""
        ),
        ("human", "Provide the cost estimation for traveling from {s} to {d}."),
    ],
    partial_variables={"output_format_instructions": parser.get_format_instructions()},
)

# Initialize Gemini AI Model-LOGIC2
chat_model = ChatGoogleGenerativeAI(api_key="your_api_key", model="gemini-2.0-flash-exp")

# Chain Components
chain = chat_template | chat_model | parser

# Streamlit UI
st.title("🚆✈️🚗 Travel Cost Estimator")
st.write("Enter your source and destination to get an approximate travel cost breakdown.")

# User Inputs
source = st.text_input("Source City", "")
destination = st.text_input("Destination City", "")

# Button to Fetch Cost Estimates
if st.button("Get Cost Estimate"):
    if source and destination:
        raw_ip = {"s": source, "d": destination}
        with st.spinner("Fetching cost estimates..."):
            try:
                # Get Response
                response = chain.invoke(raw_ip)
                st.subheader(f"The transportation cost from {source} to {destination} is given below:")
                # Display JSON Response
                st.json(response)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    else:
        st.warning("Please enter both source and destination.")
