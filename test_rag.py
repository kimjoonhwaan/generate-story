#!/usr/bin/env python3
"""
Simple test script for the RAG Story Generator

This script tests the basic functionality of the RAG system
to ensure everything is working correctly.
"""

import os
import sys
from rag_system import RAGSystem

def test_basic_functionality():
    """Test basic RAG system functionality"""
    print("🧪 Starting RAG System Basic Functionality Test\n")
    
    try:
        # Initialize RAG system (without OpenAI to avoid API key issues)
        print("1. Initializing RAG System (local mode)...")
        rag_system = RAGSystem(use_openai=False)
        print("   ✅ RAG System initialized successfully\n")
        
        # Test database stats (should be empty initially)
        print("2. Checking initial database stats...")
        stats = rag_system.get_database_stats()
        print(f"   📊 Database count: {stats['count']} documents")
        print(f"   📊 Collection name: {stats['name']}")
        print("   ✅ Database stats retrieved successfully\n")
        
        # Test file processing with sample file
        print("3. Testing file processing...")
        sample_file = "sample_stories.txt"
        
        if os.path.exists(sample_file):
            success = rag_system.add_file_to_database(sample_file)
            if success:
                print("   ✅ Sample file processed successfully")
                
                # Check updated stats
                updated_stats = rag_system.get_database_stats()
                print(f"   📊 Updated database count: {updated_stats['count']} documents")
            else:
                print("   ❌ Failed to process sample file")
                return False
        else:
            print(f"   ⚠️  Sample file '{sample_file}' not found, skipping file processing test")
        print()
        
        # Test story generation
        print("4. Testing story generation...")
        test_keywords = "magic, adventure, forest, crystal"
        print(f"   🎯 Using keywords: {test_keywords}")
        
        result = rag_system.search_and_generate_story(test_keywords, "short")
        
        if result and result.get('story'):
            print("   ✅ Story generated successfully!")
            print(f"   📚 Context documents used: {result['search_results_count']}")
            print(f"   📝 Story preview: {result['story'][:100]}...")
        else:
            print("   ❌ Failed to generate story")
            return False
        print()
        
        # Test search functionality
        print("5. Testing search functionality...")
        if os.path.exists(sample_file):
            search_results = rag_system.vector_db.search("forest magic", n_results=3)
            if search_results:
                print(f"   🔍 Found {len(search_results)} relevant documents")
                print("   ✅ Search functionality working")
            else:
                print("   ⚠️  No search results found (this might be normal)")
        else:
            print("   ⚠️  Skipping search test (no documents in database)")
        print()
        
        print("🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("- Run 'python main.py' to start the web interface")
        print("- Run 'python main.py --cli' for command-line interface")
        print("- Upload your own text files for better story context")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\nTroubleshooting:")
        print("- Make sure all dependencies are installed: pip install -r requirements.txt")
        print("- Check that you have write permissions in the current directory")
        print("- Try deleting the 'chroma_db' folder if it exists and run the test again")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("📦 Checking Dependencies...\n")
    
    required_packages = [
        'streamlit',
        'chromadb', 
        'sentence_transformers',
        'openai',
        'langchain',
        'python-dotenv',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def main():
    print("=" * 60)
    print("RAG Story Generator - Test Suite")
    print("=" * 60)
    print()
    
    # Test dependencies first
    if not test_dependencies():
        sys.exit(1)
    
    print()
    
    # Test basic functionality
    if not test_basic_functionality():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎯 Test Suite Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 