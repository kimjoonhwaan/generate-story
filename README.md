# 📚 RAG Story Generator

A Retrieval-Augmented Generation (RAG) application that creates engaging English stories based on keywords and uploaded documents. The system uses vector embeddings to find relevant context from your documents and generates creative stories using both OpenAI API and local generation methods.

## ✨ Features

- **Document Processing**: Upload and process various file formats (.txt, .csv, .pdf) into a vector database
- **Semantic Search**: Find relevant content using vector similarity search
- **Story Generation**: Create stories using keywords with two modes:
  - **OpenAI API**: High-quality stories using GPT-4
  - **Local Generation**: Template-based stories that work without API keys
- **RAG Vocabulary Restriction**: Option to use only words from uploaded documents
- **Web Interface**: Beautiful Streamlit-based GUI
- **CLI Interface**: Command-line interface for terminal users
- **Persistent Storage**: ChromaDB for storing document embeddings

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd projectRag2

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup (Optional - for OpenAI API)

For better story generation, set up your OpenAI API key:

```bash
# Copy the environment template
cp env_template.txt .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### 3. Test the Installation (Optional)

```bash
# Run the test suite to verify everything is working
python test_rag.py
```

### 4. Run the Application

#### Web Interface (Recommended)
```bash
python main.py
# or
python main.py --web
```

Then open your browser to `http://localhost:8501`

#### Command Line Interface
```bash
python main.py --cli
```

## 🎯 How to Use

### Web Interface

1. **Initialize the System**: Click "🚀 Initialize RAG System" in the sidebar
2. **Upload Documents**: Add your text files to build the knowledge base
3. **Generate Stories**: Enter keywords and select story length
4. **Enjoy**: Read and download your generated stories!

### CLI Interface

1. The system will initialize automatically
2. Follow the prompts to:
   - Type keywords for story generation
   - Use `stats` to see database information
   - Use `clear` to reset the database
   - Use `quit` to exit

## 📋 Example Usage

### Sample Keywords
- **Adventure**: `adventure, hero, quest, danger, courage`
- **Mystery**: `mystery, detective, clues, secrets, investigation`
- **Fantasy**: `magic, wizard, dragon, castle, spell`
- **Sci-Fi**: `space, robot, future, technology, alien`

### Sample Story Output
With keywords: `magic, forest, crystal, adventure`

> In a world where magic was everything, a young adventurer named Alex discovered a mysterious crystal. As Alex explored deeper into the enchanted forest, they encountered ancient spells and realized that the crystal held the key to understanding this strange place. With newfound wisdom about adventure, Alex returned home, forever changed by the journey.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Files    │───▶│  Text Processor  │───▶│   Vector DB     │
│   (.txt, .csv)  │    │  (Chunking +     │    │   (ChromaDB)    │
└─────────────────┘    │   Embedding)     │    └─────────────────┘
                       └──────────────────┘             │
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌───────▼─────────┐
│  Generated      │◀───│  Story Generator │◀───│  Search Engine  │
│    Story        │    │  (OpenAI/Local)  │    │  (Similarity)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │    Keywords     │
                       │   (User Input)  │
                       └─────────────────┘
```

## 📁 Project Structure

```
projectRag2/
├── main.py              # Main entry point
├── app.py               # Streamlit web interface
├── rag_system.py        # Main RAG system class
├── vector_db.py         # Vector database management
├── text_processor.py    # Text processing and chunking
├── story_generator.py   # Story generation logic
├── requirements.txt     # Python dependencies
├── env_template.txt     # Environment variables template
└── README.md           # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Text Processing Settings

Modify `text_processor.py` to adjust:
- `chunk_size`: Maximum characters per text chunk (default: 500)
- `chunk_overlap`: Characters overlap between chunks (default: 50)

### Vector Database Settings

Modify `vector_db.py` to adjust:
- Embedding model: Default is `sentence-transformers/all-MiniLM-L6-v2`
- Similarity metric: Default is cosine similarity

## 🛠️ Dependencies

- **streamlit**: Web interface framework
- **chromadb**: Vector database for embeddings
- **sentence-transformers**: Text embedding models
- **openai**: OpenAI API client
- **langchain**: LLM framework utilities
- **python-dotenv**: Environment variable management
- **pandas**: Data processing
- **numpy**: Numerical computations
- **PyPDF2**: PDF file processing
- **pdfplumber**: Advanced PDF text extraction

## 🔧 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found" warning**
   - This is normal if you haven't set up OpenAI API
   - The system will use local generation instead

2. **ChromaDB initialization errors**
   - Delete the `chroma_db` folder and restart
   - Make sure you have write permissions in the project directory

3. **Import errors**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Make sure you're using Python 3.8 or higher

4. **Streamlit not starting**
   - Check if port 8501 is already in use
   - Try running `streamlit run app.py` directly

### Performance Tips

- **Upload relevant documents**: The more relevant your uploaded documents, the better the story context
- **Use specific keywords**: More specific keywords lead to better stories
- **Optimal keyword count**: 3-7 keywords work best
- **File size**: Large files are automatically chunked for optimal processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Look through existing issues in the repository
3. Create a new issue with detailed information about your problem

## 🎨 Future Enhancements

- [x] Support for PDF files (completed)
- [ ] Support for DOCX and other formats
- [ ] Multiple language support
- [ ] Advanced story templates
- [ ] Story rating and feedback system
- [ ] Export to different formats (HTML, EPUB)
- [ ] Collaborative story generation
- [ ] Custom embedding models

---

**Happy Story Generating! 📖✨**