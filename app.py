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
                if st.session_state.rag_system:
                    try:
                        st.session_state.rag_system.clear_database()
                        st.session_state.uploaded_files = []
                        st.success("Database cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing database: {e}")
                        # Force reinitialize if clear fails
                        try:
                            st.session_state.rag_system = RAGSystem()
                            st.session_state.uploaded_files = []
                            st.success("Database reinitialized!")
                            st.rerun()
                        except Exception as e2:
                            st.error(f"Failed to reinitialize: {e2}")
    
    # Vocabulary viewer section
    if st.session_state.database_initialized:
        st.header("📚 RAG Vocabulary")
        
        try:
            vocabulary = st.session_state.rag_system.vector_db.get_vocabulary()
            total_words = len(vocabulary)
            
            if total_words > 0:
                # Search functionality
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_term = st.text_input(
                        "🔍 Search vocabulary",
                        placeholder="Type to search words...",
                        help="Search for specific words in the vocabulary"
                    )
                with col2:
                    show_all = st.checkbox("Show all", value=False)
                
                # Filter vocabulary based on search
                if search_term:
                    filtered_vocab = [word for word in vocabulary if search_term.lower() in word.lower()]
                    st.info(f"Found {len(filtered_vocab)} words containing '{search_term}' out of {total_words} total words")
                else:
                    filtered_vocab = vocabulary
                
                # Display options
                col1, col2, col3 = st.columns(3)
                with col1:
                    words_per_page = st.selectbox(
                        "Words per page",
                        [20, 50, 100, 200],
                        index=1,
                        help="Number of words to display per page"
                    )
                with col2:
                    sort_option = st.selectbox(
                        "Sort by",
                        ["Alphabetical", "Length (short to long)", "Length (long to short)"],
                        help="How to sort the vocabulary"
                    )
                with col3:
                    display_mode = st.selectbox(
                        "Display mode",
                        ["Grid", "List", "Table"],
                        help="How to display the words"
                    )
                
                # Sort vocabulary
                if sort_option == "Alphabetical":
                    filtered_vocab = sorted(filtered_vocab)
                elif sort_option == "Length (short to long)":
                    filtered_vocab = sorted(filtered_vocab, key=len)
                elif sort_option == "Length (long to short)":
                    filtered_vocab = sorted(filtered_vocab, key=len, reverse=True)
                
                # Pagination
                if not show_all and len(filtered_vocab) > words_per_page:
                    total_pages = (len(filtered_vocab) - 1) // words_per_page + 1
                    page = st.selectbox(
                        f"Page (1-{total_pages})",
                        range(1, total_pages + 1),
                        help=f"Navigate through {total_pages} pages of vocabulary"
                    )
                    
                    start_idx = (page - 1) * words_per_page
                    end_idx = min(start_idx + words_per_page, len(filtered_vocab))
                    display_vocab = filtered_vocab[start_idx:end_idx]
                    
                    st.info(f"Showing words {start_idx + 1}-{end_idx} of {len(filtered_vocab)}")
                else:
                    display_vocab = filtered_vocab
                    if len(filtered_vocab) > 200 and not show_all:
                        st.warning("Showing first 200 words. Check 'Show all' to see all words.")
                        display_vocab = filtered_vocab[:200]
                
                # Display vocabulary
                if display_mode == "Grid":
                    # Grid display with word length info
                    cols = st.columns(5)
                    for i, word in enumerate(display_vocab):
                        with cols[i % 5]:
                            word_length = len(word)
                            if word_length == 1:
                                st.markdown(f"🔹 **{word}** `({word_length})`")
                            elif word_length == 2:
                                st.markdown(f"🔸 **{word}** `({word_length})`")
                            elif word_length <= 5:
                                st.markdown(f"🟡 **{word}** `({word_length})`")
                            elif word_length <= 10:
                                st.markdown(f"🟢 **{word}** `({word_length})`")
                            else:
                                st.markdown(f"🔵 **{word}** `({word_length})`")
                
                elif display_mode == "List":
                    # List display with categories
                    word_categories = {
                        "1 letter": [w for w in display_vocab if len(w) == 1],
                        "2 letters": [w for w in display_vocab if len(w) == 2],
                        "3-5 letters": [w for w in display_vocab if 3 <= len(w) <= 5],
                        "6-10 letters": [w for w in display_vocab if 6 <= len(w) <= 10],
                        "11+ letters": [w for w in display_vocab if len(w) > 10]
                    }
                    
                    for category, words in word_categories.items():
                        if words:
                            with st.expander(f"{category} ({len(words)} words)"):
                                st.write(", ".join(words))
                
                elif display_mode == "Table":
                    # Table display with additional info
                    import pandas as pd
                    
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
            "Enter keywords for story generation (comma-separated):",
            placeholder="e.g., adventure, mystery, forest, magic, friendship",
            help="Enter English keywords that will be used to generate your story"
        )
    
    with col2:
        story_length = st.selectbox(
            "Story Length:",
            ["short", "medium", "long"],
            index=1,
            help="Choose the desired length of your generated story"
        )
        
        use_rag_vocab_only = st.checkbox(
            "Use only RAG vocabulary",
            value=False,
            help="생성된 스토리에서 업로드된 문서의 단어들만 사용합니다"
        )
    
    if st.button("🎭 Generate Story", type="primary", disabled=not keywords.strip()):
        generate_story(keywords, story_length, use_rag_vocab_only)
    
    # Example section
    with st.expander("💡 Examples & Tips"):
        st.markdown("""
        ### Example Keywords:
        - **Adventure**: `adventure, hero, quest, danger, courage`
        - **Mystery**: `mystery, detective, clues, secrets, investigation`
        - **Fantasy**: `magic, wizard, dragon, castle, spell`
        - **Sci-Fi**: `space, robot, future, technology, alien`
        
        ### Tips for Better Stories:
        - Use 3-7 keywords for best results
        - Mix nouns, adjectives, and action words
        - Upload relevant documents to provide better context
        - Try different story lengths to see what works best
        """)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to the database"""
    try:
        with st.spinner("Processing uploaded files..."):
            successful_files = []
            failed_files = []
            
            # Get initial stats
            initial_stats = st.session_state.rag_system.get_database_stats()
            initial_count = initial_stats['count']
            
            for uploaded_file in uploaded_files:
                # Create temporary file with proper extension
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process the file
                success = st.session_state.rag_system.add_file_to_database(tmp_file_path)
                
                if success:
                    successful_files.append(uploaded_file.name)
                else:
                    failed_files.append(uploaded_file.name)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            # Update session state
            st.session_state.uploaded_files.extend(successful_files)
            st.session_state.uploaded_files = list(set(st.session_state.uploaded_files))  # Remove duplicates
            
            # Get final stats
            final_stats = st.session_state.rag_system.get_database_stats()
            final_count = final_stats['count']
            documents_added = final_count - initial_count
            
            # Show results
            if successful_files:
                st.success(f"✅ Successfully processed {len(successful_files)} files! ({documents_added} documents added to database)")
                for filename in successful_files:
                    st.write(f"• ✅ {filename}")
                    
            if failed_files:
                st.warning(f"⚠️ Failed to process {len(failed_files)} files:")
                for filename in failed_files:
                    st.write(f"• ❌ {filename}")
                    
            if successful_files:
                st.rerun()
            elif not failed_files:
                st.error("No files were selected for processing.")
                
    except Exception as e:
        st.error(f"Error processing files: {e}")
        st.write("Please try again or check if the files are valid.")

def generate_story(keywords: str, story_length: str, use_rag_vocab_only: bool = False):
    """Generate a story based on keywords"""
    try:
        with st.spinner("🎨 Generating your story..."):
            # Add a progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulate progress
                progress_bar.progress(i + 1)
            
            result = st.session_state.rag_system.search_and_generate_story(
                keywords, 
                story_length=story_length,
                n_results=5,
                use_only_rag_vocabulary=use_rag_vocab_only
            )
        
        # Display the story
        st.success("Story generated successfully!")
        
        # Story display with enhanced information
        st.markdown("---")
        st.subheader(f"📖 Your Generated Story")
        
        # Enhanced story information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Keywords:** {result['keywords']}")
            if result.get('keywords_used'):
                keywords_used_list = result['keywords_used']
                usage_rate = result.get('keyword_usage_rate', 0)
                st.markdown(f"**Keywords used:** {len(keywords_used_list)}/{len(result['keywords'].split(','))} ({usage_rate:.0%})")
        
        with col2:
            st.markdown(f"**Generation method:** {result.get('generation_method', 'unknown').title()}")
            st.markdown(f"**Context documents:** {result['search_results_count']}")
        
        with col3:
            if result.get('vocabulary_restricted'):
                vocab_count = result.get('vocabulary_count', 0)
                st.markdown(f"**RAG vocabulary:** ✅ ({vocab_count} words)")
            else:
                st.markdown(f"**RAG vocabulary:** ❌")
        
        # Display story statistics
        story_text = result['story']
        if isinstance(story_text, str):
            word_count = len(story_text.split())
            st.markdown(f"**Word count:** {word_count} words")
        
        # Display the story in a nice container
        with st.container():
            # Ensure story_text is a string
            if isinstance(story_text, str):
                # Format the story content
                formatted_story = story_text.replace('\n', '<br>')
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid #007bff;
                        margin: 20px 0;
                        line-height: 1.6;
                        font-size: 16px;
                    ">
                        {formatted_story}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error(f"Unexpected story format: {type(story_text)}")
                st.write("Raw result:", result)
        
        # Show additional information if available
        if result.get('keywords_used'):
            with st.expander("📊 Generation Details"):
                st.write(f"**Keywords used in story:** {', '.join(result['keywords_used'])}")
                
                if result.get('context_documents_count', 0) > 0:
                    st.write(f"**Context documents used:** {result['context_documents_count']}")
                
                if result.get('vocabulary_restricted'):
                    st.write(f"**Vocabulary constraint applied:** Yes, {result.get('vocabulary_count', 0)} words available")
        
    except Exception as e:
        st.error(f"Error generating story: {e}")
        st.write("Please try again with different keywords or check your settings.")
        # Debug information
        st.write("Debug info:", str(e))

if __name__ == "__main__":
    main() 