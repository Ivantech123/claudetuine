import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from app import RAGSystem, ClaudeFineTuner

def main():
    st.set_page_config(
        page_title="Claude AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("Claude AI Assistant")
    
    # API Keys
    with st.sidebar.expander("API Keys"):
        api_key = st.text_input("Anthropic API Key", type="password")
        voyage_key = st.text_input("Voyage API Key", type="password")
        pinecone_key = st.text_input("Pinecone API Key", type="password")
        
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if voyage_key:
            os.environ["VOYAGE_API_KEY"] = voyage_key
        if pinecone_key:
            os.environ["PINECONE_API_KEY"] = pinecone_key
    
    # Mode selection
    mode = st.sidebar.radio("Select Mode", ["Fine-tuning", "RAG"])
    
    if mode == "Fine-tuning":
        st.header("Fine-tuning Mode")
        
        # Initialize fine-tuner
        tuner = ClaudeFineTuner()
        
        # File upload
        uploaded_file = st.file_uploader("Upload training data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(data.head())
            
            # Train/Test split
            test_size = st.slider("Test set size (%)", 0, 100, 20)
            test_size = test_size / 100
            
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating model..."):
                    # Split data
                    test_data = data.sample(frac=test_size, random_state=42)
                    
                    # Evaluate
                    results = tuner.evaluate_model(test_data)
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Calculate and display metrics
                    exact_matches = sum(row['expected'] == row['predicted'] for row in results)
                    accuracy = exact_matches / len(results)
                    st.metric("Exact Match Accuracy", f"{accuracy:.2%}")
                    
                    # Visualize results
                    fig = px.histogram(results_df, x='expected')
                    st.plotly_chart(fig)
                    
    else:  # RAG Mode
        st.header("RAG Mode")
        
        # Initialize RAG system
        rag = RAGSystem(
            pinecone_api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'doc', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            documents = []
            for file in uploaded_files:
                content = file.read().decode('utf-8')
                documents.append({
                    'title': file.name,
                    'content': content,
                    'source': 'upload',
                    'timestamp': datetime.now().isoformat()
                })
            
            if st.button("Add to Knowledge Base"):
                with st.spinner("Adding documents..."):
                    results = rag.batch_add_documents(documents)
                    st.success(f"Added {results['success']} documents successfully!")
                    if results['failed'] > 0:
                        st.error(f"Failed to add {results['failed']} documents")
                        
        # Clear knowledge base button
        if st.button("Clear Knowledge Base"):
            if rag.clear_knowledge_base():
                st.success("Knowledge base cleared successfully!")
            else:
                st.error("Failed to clear knowledge base")
                
        # Query interface
        st.subheader("Ask Questions")
        query = st.text_input("Enter your question:")
        
        if query:
            with st.spinner("Searching and generating answer..."):
                result = rag.query(query)
                
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    st.write("Answer:", result["answer"])
                    
                    with st.expander("View Sources"):
                        for doc in result["sources"]:
                            st.markdown(f"""
                            **Title:** {doc['title']}
                            **Score:** {doc['score']:.2f}
                            **Content:** {doc['content'][:200]}...
                            """)

if __name__ == "__main__":
    main()
