import streamlit as st

page_bg_img = """
<style>
{
background-image: url("https://img.freepik.com/free-vector/classic-vintage-rays-sunburst-retro-background_1017-33769.jpg?w=996&t=st=1697542768~exp=1697543368~hmac=93170a114e29c56f767ccc1624fe66f784f9424ea5d48acb34cfd2f001833f6a");
background-size: cover;
}
"""


st.markdown(page_bg_img, unsafe_allow_html=True)
)

# The rest of your Streamlit app goes here
st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")
