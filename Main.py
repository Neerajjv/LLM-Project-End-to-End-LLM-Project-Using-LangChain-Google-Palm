import streamlit as st
from HELPER import get_qa_chain, create_vector_db

st.title("Your Data Science Helper 📖💡🔍 ")
st.title("zieganpalg")
image = "1.jpg"
st.image(image, caption=None, width=100, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])