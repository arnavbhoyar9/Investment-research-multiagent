from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
search_tool = SerperDevTool()
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    

    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}

configs = {}

for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

agents_config = configs['agents']
tasks_config = configs['tasks']


research_analyst = Agent(
    llm=llm,
    role=agents_config['research_analyst']['role'],
    goal=agents_config['research_analyst']['goal'],
    backstory=agents_config['research_analyst']['backstory'],
    tools=[search_tool],
    verbose=True
)

investment_analyst = Agent(
    llm=llm,
    role=agents_config['investment_analyst']['role'],
    goal=agents_config['investment_analyst']['goal'],
    backstory=agents_config['investment_analyst']['backstory'],
    verbose=True
)

memo_writer = Agent(
    llm=llm,
    role=agents_config['memo_writer']['role'],
    goal=agents_config['memo_writer']['goal'],
    backstory=agents_config['memo_writer']['backstory'],
    verbose=True
)
research_task = Task(
    description=tasks_config['research_task']['description'],
    expected_output=tasks_config['research_task']['expected_output'],
    agent=research_analyst
)

analysis_task = Task(
    description=tasks_config['analysis_task']['description'],
    expected_output=tasks_config['analysis_task']['expected_output'],
    agent=investment_analyst,
    context=[research_task]
)

memo_task = Task(
    description=tasks_config['memo_task']['description'],
    expected_output=tasks_config['memo_task']['expected_output'],
    agent=memo_writer,
    context=[research_task, analysis_task]
)
crew = Crew(
    agents=[
        research_analyst,
        investment_analyst,
        memo_writer
    ],
    tasks=[
        research_task,
        analysis_task,
        memo_task
    ],
    verbose=True
)
result = crew.kickoff(
    inputs={
        "company": "NVIDIA"
    }
)

print(result)


print("Agents created successfully!")
print(research_analyst.role)
print(investment_analyst.role)
print(memo_writer.role)
print("\nTasks created successfully!")
print(research_task.description[:50])
print(analysis_task.description[:50])
print(memo_task.description[:50])
print("\nCrew created successfully!")