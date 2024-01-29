from langchain.prompts import PromptTemplate

# https://python.langchain.com/docs/modules/memory/conversational_customization
system_prompt_template = PromptTemplate(
    input_variables=['history', 'input'],
    template="""You are an English Teacher and you have to create a conversation and correct the user message if needed. Respond with no more than 100 words.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
    """
    )