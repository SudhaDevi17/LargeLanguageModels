
from langchain.agents import load_tools  # This will allow us to load tools we need
from langchain.agents import initialize_agent
from langchain.agents import (
    AgentType,
)

from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import matplotlib.pyplot as plt
from langchain.llms import OpenAI


# For OpenAI we'll use the default model for DaScie
llm = OpenAI()
tools = load_tools(["wikipedia", "serpapi", "python_repl", "terminal"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


dataset = pd.read_csv("../input/mock_data.csv")
# world_data
agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0), dataset, verbose=True
)
# Let's see how well DaScie does on a simple request.
agent.run("Analyze this data, tell me any interesting trends. Make some pretty plots.")

agent.run(
    "Train a random forest regressor to predict salary using the most important features. Show me the what variables are most influential to this model"
)