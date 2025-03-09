from swarm import Swarm, Agent
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')

class RAG:
    def __init__(self, model="gemini/gemini-2.0-pro-exp-02-05", functions=[]) -> None:
        self.model = model
        self.functions = functions
        
    def llm(self, messages, context) -> str:
        client = Swarm()
        
        agent = Agent(
            name="Agent",
            model=self.model,
            instructions=f"""You are a knowledgeable and reliable RAG assistant.  
            Answer the user's question accurately and concisely using the given information.  
            Maintain the original language of the question without explicitly stating that your response is based on provided knowledge.  
            
            ### Mandatory Presentation Guidelines (Always Follow These Rules):
            - **Always ensure responses are visually structured and easy to read**.  
            - Use **bold** or *italic* text to highlight key information.  
            - Organize content into **bullet points** or **numbered lists** for clarity.  
            - Use **tables** to present comparisons or structured data effectively.  
            - Provide **examples** whenever possible to illustrate concepts clearly.  
            - Break down long explanations into **short paragraphs** for readability.  
            - If relevant, use **headings and subheadings** to structure the response.
            
            ### Relevant Information:  
            {context}""",
            
            functions=self.functions,
            model_config={
                    "temperature": 0,
                    }
            )
        
        response = client.run(
            agent=agent,
            messages=messages,
        )

        return response.messages[-1]["content"]
    

    
