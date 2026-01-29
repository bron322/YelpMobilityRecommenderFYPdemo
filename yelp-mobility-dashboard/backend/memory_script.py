try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema import StrOutputParser
    print("SUCCESS: All libraries are installed correctly!")
except ImportError as e:
    print(f"ERROR: Missing library -> {e}")