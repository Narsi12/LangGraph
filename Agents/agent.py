import os
from datetime import datetime
from typing import Dict, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, MessagesState


# ============================================================
# 1. Environment and LLM setup
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY was not found. Add it to your .env file."
    )
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# 2. Shared state
# ============================================================
class SupervisorState(MessagesState):
    next_agent: str
    research_data: str
    analysis: str
    final_report: str
    task_complete: bool
    current_task: str


# ============================================================
# 3. Supervisor chain
# ============================================================

def create_supervisor_chain():
    supervisor_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a supervisor managing three agents:
1. Researcher: gathers background information.
2. Analyst: analyzes the research and identifies insights.
3. Writer: creates the final report.

Choose the next agent based on the current state.
Follow this order:
- If research is missing, choose researcher.
- If research exists but analysis is missing, choose analyst.
- If analysis exists but the report is missing, choose writer.
- If the report exists, respond with DONE.

Respond with ONLY one word: researcher, analyst, writer, or DONE.

Current state:
- Has research data: {has_research}
- Has analysis: {has_analysis}
- Has report: {has_report}
"""
        ),
        ("human", "Original user task: {task}"),
    ])

    return supervisor_prompt | llm


# ============================================================
# 4. Supervisor agent
# ============================================================

def supervisor_agent(state: SupervisorState) -> Dict:
    # Preserve the original task instead of using the latest agent message.
    task = state.get("current_task", "")

    if not task:
        for message in state.get("messages", []):
            if isinstance(message, HumanMessage):
                task = str(message.content)
                break

    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))

    decision = create_supervisor_chain().invoke({
        "task": task or "No task supplied",
        "has_research": has_research,
        "has_analysis": has_analysis,
        "has_report": has_report,
    })

    decision_text = decision.content.strip().lower()
    print(f"\n[Supervisor decision] {decision_text}")

    # Enforce the workflow order even if the LLM returns an unexpected choice.
    if has_report or "done" in decision_text:
        next_agent = "end"
        supervisor_msg = "Supervisor: All tasks complete."
    elif not has_research:
        next_agent = "researcher"
        supervisor_msg = "Supervisor: Assigning task to Researcher."
    elif not has_analysis:
        next_agent = "analyst"
        supervisor_msg = "Supervisor: Assigning task to Analyst."
    else:
        next_agent = "writer"
        supervisor_msg = "Supervisor: Assigning task to Writer."

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task,
    }


# ============================================================
# 5. Researcher agent
# ============================================================

def researcher_agent(state: SupervisorState) -> Dict:
    task = state.get("current_task", "Research topic")
    print("researcher_agent task", task)

    research_prompt = f"""Act as a research specialist.

Research task: {task}

Provide:
1. Key facts and background
2. Important trends and developments
3. Relevant statistics, with caveats if uncertain
4. Examples or case studies
5. Potential benefits and risks, where relevant

Be structured and concise. Do not invent citations or claim you performed
live web searches.
"""

    result = llm.invoke([HumanMessage(content=research_prompt)])
    research_data = str(result.content)

    print("\n[Researcher] Research completed.")

    return {
        "messages": [
            AIMessage(content="Researcher: Research completed.")
        ],
        "research_data": research_data,
        "next_agent": "supervisor",
    }


# ============================================================
# 6. Analyst agent
# ============================================================

def analyst_agent(state: SupervisorState) -> Dict:
    task = state.get("current_task", "")
    print("analyst_agent task",task)
    research_data = state.get("research_data", "")

    analysis_prompt = f"""Act as an analyst.

Original task: {task}

Research data:
{research_data}

Analyze the research and provide:
1. Key insights and patterns
2. Benefits and risks
3. Strategic implications
4. Risks, limitations, and uncertainties
5. Practical recommendations

Base your analysis on the supplied research. Clearly distinguish evidence
from assumptions.
"""

    result = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = str(result.content)

    print("\n[Analyst] Analysis completed.")

    return {
        "messages": [
            AIMessage(content="Analyst: Analysis completed.")
        ],
        "analysis": analysis,
        "next_agent": "supervisor",
    }


# ============================================================
# 7. Writer agent
# ============================================================

def writer_agent(state: SupervisorState) -> Dict:
    task = state.get("current_task", "")
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")

    writing_prompt = f"""Act as a professional report writer.

Original task: {task}

Research findings:
{research_data[:12000]}

Analysis:
{analysis[:12000]}

Write a professional report with these sections:
1. Executive Summary
2. Key Findings
3. Analysis and Insights
4. Risks and Limitations
5. Recommendations
6. Conclusion

Do not invent sources or unsupported statistics.
"""

    result = llm.invoke([HumanMessage(content=writing_prompt)])
    report = str(result.content)

    final_report = f"""FINAL REPORT
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Topic: {task}
{'=' * 60}

{report}

{'=' * 60}
Report generated by the LangGraph multi-agent system using Groq.
"""

    print("\n[Writer] Final report completed.")

    return {
        "messages": [
            AIMessage(content="Writer: Final report completed.")
        ],
        "final_report": final_report,
        "next_agent": "end",
        "task_complete": True,
    }


# ============================================================
# 8. Router
# ============================================================

def router(
    state: SupervisorState,
) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    if state.get("task_complete", False):
        return END

    next_agent = state.get("next_agent", "supervisor")

    if next_agent == "end":
        return END

    if next_agent in {"supervisor", "researcher", "analyst", "writer"}:
        return next_agent

    return "supervisor"


# ============================================================
# 9. Build and compile the workflow
# ============================================================
workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)

workflow.set_entry_point("supervisor")

for node_name in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node_name,
        router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END,
        },
    )

graph = workflow.compile()


# ============================================================
# 10. Run the workflow
# ============================================================

if __name__ == "__main__":
    user_question = "What are the benefits and risks of AI in healthcare?"
    response = graph.invoke({
        "messages": [HumanMessage(content=user_question)],
        "current_task": user_question,
        "next_agent": "supervisor",
        "research_data": "",
        "analysis": "",
        "final_report": "",
        "task_complete": False,
    })
    

    print("\n\n")
    print(response.get("final_report", "No final report was generated."))
