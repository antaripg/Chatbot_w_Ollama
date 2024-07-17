import streamlit as st 
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)



# Initialize chat history in session state if not present
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Function to stream chat response based on selected model
def stream_chat(model, messages):
    try:
        # Initialize the language model with a timeout
        llm = Ollama(model=model, request_timeout=120.0)
        # Stream chat responses from the model
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()

        # Append each piece of the response to the output
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        # Log the interaction details
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        # Log the error and re-raise any errors that occur
        logging.error(f"Error during streaming: {str(e)}")
        raise e
    
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
# Streamlit MAIN Function
def main():
    # st.set_page_config(page_title=":robot: OllamaChat")
    st.title("Local Chat")
    logging.info("App Started")

    # Sidebar for model selection
    model = st.sidebar.selectbox("Chose a Model", ["llama2", "mistral:v0.2", "phi3:medium"])
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    logging.info(f"Model Selected: {model}")

    # Prompt for user input and save to chat history
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        # Display the user's query
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Generate a new response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != 'assistant':
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating Response")

                with st.spinner("Writing..."):
                    try:
                        # Prepare messages for the LLM and stream the response
                        messages = [ChatMessage(role=msg['role'], 
                                                content=msg['content']) for msg in st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message} \n\nDuration: {duration}"
                        st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                        st.write(f"Duration: {duration: .2f} seconds")
                        logging.info(f"Reponse: {response_message}, Duration: {duration}")
                    except Exception as e:
                        # Handle errors and display an error message
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occured while generating the response")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
