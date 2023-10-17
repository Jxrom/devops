import streamlit as st

# Set the background image using CSS
background_image_url = "https://img.freepik.com/free-vector/classic-vintage-rays-sunburst-retro-background_1017-33769.jpg?w=996&t=st=1697542768~exp=1697543368~hmac=93170a114e29c56f767ccc1624fe66f784f9424ea5d48acb34cfd2f001833f6a"  # Replace with the URL of your background image

st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-repeat: no-repeat;
        }}
    </style>
    """,
    unsafe_allow_html=True  # Enable HTML in Markdown
)

# Set the page configuration (can only be called once, and must be the first Streamlit command)
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

# The rest of your Streamlit app goes here
st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
