from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ModelInfo, ModelFamily
from Tools import arxiv_search 

class AgentTeam: 
    def __init__(self): 
        '''
        Initializes the agent team, defining all agents and the group chat.
        '''
        
        researcher_model_client = OllamaChatCompletionClient(
            model="granite3.3:2b",
            client_host="http://localhost:11434",
            native_tool_calls=False,
            hide_tools=True,
            model_info=ModelInfo(
                vision=False,
                function_calling=True,
                json_output=True,
                family=ModelFamily.UNKNOWN,
                structured_output=True
            )
        )


        writer_model_client = OllamaChatCompletionClient(
            model="granite3.3:8b",
            client_host="http://localhost:11434",
            model_info=ModelInfo(
                vision=False,
                function_calling=False,
                json_output=False,
                family=ModelFamily.UNKNOWN,
                structured_output=False
            )
        )

        self.researcher = AssistantAgent(
            name="Researcher",
            model_client=researcher_model_client,
            tools=[arxiv_search],
            reflect_on_tool_use=False,
            model_client_stream=True,
            system_message=(
                'given a user topic, think of the best arXiv query. When the tool returns'
                'choose exactly the number of papers requested and pass them as JSON with all information '
                'to the Reviewer, make sure to return the PDF_URL field'
            )
        )


        self.writer = AssistantAgent(
            name="Writer",
            model_client=writer_model_client, 
            model_client_stream=True,
            system_message='''
                You are an expert academic writer, write a cohesive and insightful literature 
                review based exclusively on the provided papers information.
                You will perform a  literature review following these steps in order:

                1.  Introduction & Scope:
                    - Start with a concise 2-3 sentence introduction that defines the research area covered by the papers.
                    - Then, list the reviewed papers with their titles, authors and paper_url.

                2.  Thematic Synthesis (The Core of Your Task):
                    - Identify 2-3 central themes that emerge from the collection of abstracts make references of the papers you are taking information from.
                    - For each theme, including the reference of the paper, create a separate paragraph. In each paragraph:
                        - Clearly state the theme.
                        - Detail how each relevant paper contributes to, defines, or challenges the theme asked for the user.
                        - Compare and contrast the papers approaches and findings within the theme. Use phrases like "While Paper A focuses on...,
                          Paper B offers a contrasting view by...".

                3.  Methodological Overview & Limitations :
                    - Briefly summarize the methodologies mentioned in the abstracts, including the reference to the papers 
                    (e.g., "The reviewed studies employ a mix of qualitative analysis, 
                    machine learning models, and user surveys...").
                    - Point out any potential limitations or gaps that can be inferred *solely from the abstracts* (e.g., "A potential 
                    gap appears to be the lack of focus on long-term user studies...").

                4.  Conclusion and Future Directions:
                    - Conclude with a powerful 1-2 sentence summary of the current state of research based on these papers.
                    - Suggest a key direction for future research that logically follows from your analysis.

                    '''
                )
        
        self.reviewer = AssistantAgent(
            name="Reviewer",
            model_client=writer_model_client,
            model_client_stream=True,
            system_message='''
            You are a reviewer of academic papers retrieved by the Researcher Agent. Your goal is to check if the selected papers are aligned with the 
            user's query and if they cover the key points of interest.Keep your response clear and focused, using only the information available in the abstracts. 
            Select the most relevant papers based on the user’s query and return exactly the number requested. Each paper must be formatted with its title, authors, 
            published date, summary, and PDF URL.

            Given:
            - A user topic or research query.
            - A list of selected papers in the format:
            Title: ,
                PDF URL: ,
                Authors: ,
                Published: , 
                Summary: , 
                

            You must:
            1. Assess Relevance:
            -Check if each paper is clearly connected to the user query.
            -Identify each paper that are off-topic or only loosely related.

            2. Coverage Check:
            - Briefly state if the collection of papers addresses:
                - The main topic or question.
                - Sub-topics or dimensions (e.g., methodology, application, theory).
            Note if any important angles might be missing.

            3.Output:
            - Write a short paragraph (3–5 sentences) evaluating the overall match between the papers and the user query and attention key points.

            '''
        )

        # Configuration of the team
        self.team = RoundRobinGroupChat([self.researcher, self.reviewer, self.writer], max_turns=3)

    
   

    async def run_chat(self, task: str) -> str:
        """
        Runs the agent team and processes the event stream to build a clean log,
        correctly parsing all event data structures.
        """
        stream = self.team.run_stream(task=task)

        conversation_flow = [{"source": "User", "content": task, "type": "text"}]

        async for event in stream:
            event_type = getattr(event, 'type', '')
            source = getattr(event, 'source', 'system')

            if event_type == 'ToolCallRequestEvent':
                tool_calls = getattr(event, 'content', [])
                for call in tool_calls:
                    conversation_flow.append({
                        "source": source,
                        "type": "tool_request",
                        "tool_name": call.name,
                        "arguments": call.arguments
                    })

            elif event_type == 'ToolCallExecutionEvent':
                conversation_flow.append({
                    "source": source,
                    "type": "tool_execution"
                })

            elif event_type == 'ModelClientStreamingChunkEvent':
                content_chunk = getattr(event, 'content', '')
                found = False
                for i in range(len(conversation_flow) - 1, -1, -1):
                    if conversation_flow[i].get("source") == source and conversation_flow[i].get("type") == "text":
                        conversation_flow[i]["content"] += content_chunk
                        found = True
                        break
                if not found:
                    conversation_flow.append({
                        "source": source,
                        "content": content_chunk,
                        "type": "text"
                    })

        formatted_log = []
        for item in conversation_flow:
            item_type = item["type"]
            source = item["source"]

            if item_type == "text":
                content = item.get("content", "").strip()
                if content:
                    formatted_log.append(f"**{source.title()}**:\n\n{content}")

            elif item_type == "tool_request":
                tool_name = item["tool_name"]
                arguments = item["arguments"]
                formatted_log.append(
                    f"🛠️ **Tool Call:** `{source}` is planning to call `{tool_name}` with arguments:\n"
                    f"```json\n{arguments}\n```"
                )

            elif item_type == "tool_execution":
                formatted_log.append("✅ **Tool Result:** Data received successfully.")

        return "\n\n".join(formatted_log)
