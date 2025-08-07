import streamlit as st
import os
import tempfile
import pandas as pd
from rag_system import RAGSystem
from typing import List
import time

# Configure page
st.set_page_config(
    page_title="RAG Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'database_initialized' not in st.session_state:
    st.session_state.database_initialized = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def initialize_rag_system(use_openai: bool = True):
    """Initialize the RAG system"""
    try:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = RAGSystem(use_openai=use_openai)
            st.session_state.database_initialized = True
        st.success("RAG system initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return False

def main():
    st.title("📚 RAG Story Generator")
    st.markdown("""
    Welcome to the **RAG Story Generator**! This application uses Retrieval-Augmented Generation 
    to create engaging English stories based on your keywords and uploaded documents.
    
    ### How it works:
    1. **Upload Files**: Add text documents to build your knowledge base
    2. **Enter Keywords**: Provide English keywords for story generation
    3. **Generate Story**: Let AI create a unique story using your keywords and document context
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API option
        use_openai = st.checkbox(
            "Use OpenAI API (requires API key)", 
            value=True,
            help="Enable this to use OpenAI's GPT for better story generation. Requires OPENAI_API_KEY in environment."
        )
        
        if use_openai:
            api_key_status = os.getenv('OPENAI_API_KEY')
            if api_key_status:
                st.success("✅ OpenAI API key found")
            else:
                st.warning("⚠️ OpenAI API key not found. Will use local generation.")
                st.info("To use OpenAI, create a .env file with: OPENAI_API_KEY=your_key_here")
        
        # Initialize system button
        if st.button("🚀 Initialize RAG System", type="primary"):
            initialize_rag_system(use_openai)
        
        # Database stats
        if st.session_state.database_initialized:
            st.header("📊 Database Stats")
            try:
                stats = st.session_state.rag_system.get_database_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Documents in Database", stats['count'])
                with col2:
                    vocab_count = len(st.session_state.rag_system.vector_db.get_vocabulary())
                    st.metric("Vocabulary Words", vocab_count)
                    
                # Show recently uploaded files
                if st.session_state.uploaded_files:
                    with st.expander("📁 Uploaded Files"):
                        for filename in st.session_state.uploaded_files:
                            st.write(f"• {filename}")
                            
            except Exception as e:
                st.error(f"Could not fetch database stats: {e}")
                # Try to reinitialize if there's a problem
                if st.button("🔄 Reinitialize Database"):
                    try:
                        st.session_state.rag_system = RAGSystem()
                        st.success("Database reinitialized!")
                        st.rerun()
                    except Exception as e2:
                        st.error(f"Failed to reinitialize: {e2}")
            
            if st.button("🗑️ Clear Database", type="secondary"):
                try:
                    st.session_state.rag_system.clear_database()
                    st.session_state.uploaded_files = []
                    st.success("Database cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear database: {e}")
        
        # RAG Vocabulary section
        if st.session_state.database_initialized:
            st.header("📖 RAG Vocabulary")
            try:
                vocabulary = st.session_state.rag_system.vector_db.get_vocabulary()
                
                if vocabulary:
                    # Search functionality
                    search_term = st.text_input("🔍 Search vocabulary", placeholder="Enter word to search...")
                    
                    # Filter vocabulary based on search
                    if search_term:
                        display_vocab = [word for word in vocabulary if search_term.lower() in word.lower()]
                    else:
                        display_vocab = vocabulary
                    
                    # Display options
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        sort_option = st.selectbox(
                            "Sort by",
                            ["Alphabetical", "Length (short to long)", "Length (long to short)"]
                        )
                    with col2:
                        display_mode = st.selectbox(
                            "Display mode",
                            ["Grid", "List", "Table"]
                        )
                    
                    # Apply sorting
                    if sort_option == "Alphabetical":
                        display_vocab = sorted(display_vocab)
                    elif sort_option == "Length (short to long)":
                        display_vocab = sorted(display_vocab, key=len)
                    elif sort_option == "Length (long to short)":
                        display_vocab = sorted(display_vocab, key=len, reverse=True)
                    
                    # Pagination
                    items_per_page = 50
                    total_pages = (len(display_vocab) - 1) // items_per_page + 1
                    
                    if total_pages > 1:
                        page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
                        start_idx = page * items_per_page
                        end_idx = start_idx + items_per_page
                        display_vocab = display_vocab[start_idx:end_idx]
                    
                    # Display vocabulary
                    if display_mode == "Grid":
                        # Grid display
                        cols = st.columns(5)
                        for i, word in enumerate(display_vocab):
                            col_idx = i % 5
                            with cols[col_idx]:
                                st.write(f"`{word}`")
                    
                    elif display_mode == "List":
                        # List display with categorization
                        categories = {
                            "Articles": [w for w in display_vocab if w in ["a", "an", "the"]],
                            "Prepositions": [w for w in display_vocab if w in ["in", "on", "at", "by", "for", "with", "to", "from"]],
                            "Pronouns": [w for w in display_vocab if w in ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]],
                            "Auxiliary Verbs": [w for w in display_vocab if w in ["am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "does", "did"]],
                            "Content Words": [w for w in display_vocab if w not in ["a", "an", "the", "in", "on", "at", "by", "for", "with", "to", "from", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "does", "did"]]
                        }
                        
                        for category, words in categories.items():
                            if words:
                                with st.expander(f"{category} ({len(words)} words)"):
                                    st.write(", ".join(words))
                
                elif display_mode == "Table":
                    # Table display with additional info
                    
                    word_data = []
                    for word in display_vocab:
                        word_data.append({
                            "Word": word,
                            "Length": len(word),
                            "Type": "Article" if word in ["a", "an", "the"] else
                                   "Preposition" if word in ["in", "on", "at", "by", "for", "with", "to", "from"] else
                                   "Pronoun" if word in ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"] else
                                   "Verb" if word in ["am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "does", "did"] else
                                   "Content"
                        })
                    
                    df = pd.DataFrame(word_data)
                    st.dataframe(df, use_container_width=True)
                
                # Word statistics
                with st.expander("📊 Vocabulary Statistics"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Words", len(vocabulary))
                    with col2:
                        one_letter = len([w for w in vocabulary if len(w) == 1])
                        st.metric("1-letter words", one_letter)
                    with col3:
                        two_letter = len([w for w in vocabulary if len(w) == 2])
                        st.metric("2-letter words", two_letter)
                    with col4:
                        long_words = len([w for w in vocabulary if len(w) > 10])
                        st.metric("10+ letter words", long_words)
                    
                    # Word length distribution
                    length_counts = {}
                    for word in vocabulary:
                        length = len(word)
                        length_counts[length] = length_counts.get(length, 0) + 1
                    
                    if length_counts:
                        st.subheader("Word Length Distribution")
                        length_df = pd.DataFrame([
                            {"Length": length, "Count": count} 
                            for length, count in sorted(length_counts.items())
                        ])
                        st.bar_chart(length_df.set_index("Length"))
                
                # Export functionality
                if st.button("📥 Export Vocabulary"):
                    vocab_text = "\n".join(vocabulary)
                    st.download_button(
                        label="Download as TXT",
                        data=vocab_text,
                        file_name="rag_vocabulary.txt",
                        mime="text/plain"
                    )
                    
            else:
                st.info("No vocabulary words found. Upload some documents first!")
                
        except Exception as e:
            st.error(f"Error loading vocabulary: {e}")

    # Main content
    if not st.session_state.database_initialized:
        st.info("👈 Please initialize the RAG system using the sidebar to get started.")
        return
    
    # File upload section
    st.header("📁 Upload Documents")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files to add to your knowledge base",
            type=['txt', 'csv', 'pdf'],
            accept_multiple_files=True,
            help="Upload .txt, .csv, or .pdf files containing the text you want to use for story generation context."
        )
    
    with col2:
        if uploaded_files and st.button("📤 Process Files", type="primary"):
            process_uploaded_files(uploaded_files)
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("✅ Successfully Processed Files:")
        for filename in st.session_state.uploaded_files:
            st.text(f"• {filename}")
    
    # Story generation section
    st.header("✨ Generate Story")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        keywords = st.text_input(
            "Enter keywords for your story",
            placeholder="e.g., adventure, friendship, school",
            help="Enter English keywords separated by commas. The story will be generated based on these keywords."
        )
    
    with col2:
        story_length = st.selectbox(
            "Story length",
            ["short", "medium", "long"],
            help="Choose the desired length of your story."
        )
    
    # RAG vocabulary restriction option
    use_rag_vocab_only = st.checkbox(
        "Use only RAG vocabulary",
        value=False,
        help="When enabled, the story will be generated using only words from your uploaded documents (plus essential grammar words)."
    )
    
    if st.button("🎯 Generate Story", type="primary"):
        if keywords.strip():
            generate_story(keywords, story_length, use_rag_vocab_only)
        else:
            st.warning("Please enter keywords for story generation.")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the RAG system"""
    if not st.session_state.database_initialized:
        st.error("Please initialize the RAG system first.")
        return
    
    try:
        with st.spinner("Processing files..."):
            processed_files = []
            
            for uploaded_file in uploaded_files:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Add file to RAG system
                    st.session_state.rag_system.add_file_to_database(tmp_file_path)
                    processed_files.append(uploaded_file.name)
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
            
            # Update session state
            st.session_state.uploaded_files.extend(processed_files)
            
            st.success(f"Successfully processed {len(processed_files)} file(s)!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error processing files: {e}")

def generate_story(keywords: str, story_length: str, use_rag_vocab_only: bool = False):
    """Generate a story based on keywords and settings"""
    if not st.session_state.database_initialized:
        st.error("Please initialize the RAG system first.")
        return
    
    try:
        with st.spinner("Generating your story..."):
            result = st.session_state.rag_system.search_and_generate_story(
                keywords=keywords,
                story_length=story_length,
                use_rag_vocab_only=use_rag_vocab_only
            )
            
            if result and result.get('story'):
                # Display the generated story
                st.subheader("📖 Generated Story")
                
                # Story text
                story_text = result['story']
                st.write(story_text)
                
                # Story metadata
                with st.expander("📊 Story Details"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Word Count", result.get('word_count', 0))
                    with col2:
                        st.metric("Keywords Used", f"{result.get('keywords_used', 0)}/{len(keywords.split(','))}")
                    with col3:
                        st.metric("Generation Method", result.get('generation_method', 'Unknown'))
                    
                    # Context documents
                    if result.get('context_documents'):
                        st.write("**Context Documents:**")
                        for i, doc in enumerate(result['context_documents'][:3], 1):
                            st.write(f"{i}. {doc[:100]}...")
                
                # Vocabulary analysis (if RAG vocabulary was used)
                if use_rag_vocab_only and result.get('vocabulary_analysis'):
                    st.subheader("📝 Vocabulary Analysis Results")
                    
                    analysis = result['vocabulary_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Words", analysis.get('total_words', 0))
                    with col2:
                        st.metric("RAG Usage", f"{analysis.get('rag_usage_percentage', 0):.1f}%")
                    with col3:
                        st.metric("Non-RAG Words", analysis.get('non_rag_count', 0))
                    
                    # Non-RAG words by category
                    if analysis.get('non_rag_words'):
                        st.write("**Non-RAG Words by Category:**")
                        
                        categories = ['nouns', 'verbs', 'adjectives', 'adverbs', 'others']
                        for category in categories:
                            words = analysis['non_rag_words'].get(category, [])
                            if words:
                                with st.expander(f"{category.title()} ({len(words)} words)"):
                                    st.write(", ".join(words))
            else:
                st.error("Failed to generate story. Please try again.")
                
    except Exception as e:
        st.error(f"Error generating story: {e}")

if __name__ == "__main__":
    main() 