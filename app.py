import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from Llama 2 model
def getLlamaResponse(blog_style, input_text, num_of_words):
    try:
        # Calling the Llama2 model
        llm = CTransformers(
            model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
            model_type="llama",
            config={"max_new_tokens": 256, "temperature": 0.01}
        )
        
        # Prompt Template
        template = """
        Write a blog for {blog_style} job profile on the topic {input_text} within {num_of_words} words.
        """
        
        prompt = PromptTemplate(
            input_variables=["blog_style", "input_text", "num_of_words"],
            template=template
        )
        
        # Generate the response from the Llama 2 model
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, num_of_words=num_of_words))

        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Streamlit UI
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs 📝")

input_text = st.text_input("Enter the Blog Topic:")

# Creating 2 more columns for the fields
col1, col2 = st.columns([1, 1])

with col1:
    num_of_words = st.text_input("Number of Words:", "500")
with col2:
    blog_style = st.selectbox("Writing the Blog for:", 
                              ('Researchers', 'Data Scientists', 'Common People'), index=0)
    
submit = st.button("Generate")

if submit:
    if input_text and num_of_words:
        response = getLlamaResponse(blog_style, input_text, num_of_words)
        st.write(response)
    else:
        st.error("Please enter the blog topic and the number of words.")
