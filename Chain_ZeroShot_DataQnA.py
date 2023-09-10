import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langsmith import Client

df = pd.read_csv('spocs_data.csv')

llm = ChatOpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df, agent_type=AgentType.OPENAI_FUNCTIONS)

client = Client()
def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)

query_text = 'Who was in cabin C128?'
result = None

response = agent({"input": query_text}, include_run_info=True)
result = response["output"]
run_id = response["__run"].run_id
print(result)
