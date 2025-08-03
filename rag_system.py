from vector_db import VectorDB
from text_processor import TextProcessor
from story_generator import StoryGenerator
from typing import List, Dict
import os

class RAGSystem:
    def __init__(self, use_openai: bool = True):
        """
        Initialize the RAG System
        
        Args:
            use_openai: Whether to use OpenAI for story generation
        """
        self.vector_db = VectorDB()
        self.text_processor = TextProcessor()
        self.story_generator = StoryGenerator(use_openai=use_openai)
        
        print("RAG System initialized successfully!")
        
    def add_file_to_database(self, file_path: str) -> bool:
        """
        Process a file and add it to the vector database
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
            
            # Process the file
            result = self.text_processor.process_file(file_path)
            
            if not result['chunks']:
                print(f"No content found in file: {file_path}")
                return False
            
            # Add to vector database
            self.vector_db.add_documents(result['chunks'], result['metadata'])
            
            print(f"Successfully added {len(result['chunks'])} chunks from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return False
    
    def add_multiple_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Add multiple files to the database
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary with file paths as keys and success status as values
        """
        results = {}
        
        for file_path in file_paths:
            results[file_path] = self.add_file_to_database(file_path)
        
        successful = sum(1 for success in results.values() if success)
        print(f"\nProcessed {len(file_paths)} files. {successful} successful, {len(file_paths) - successful} failed.")
        
        return results
    
    def search_and_generate_story(self, keywords: str, story_length: str = "medium", 
                                 n_results: int = 5, use_only_rag_vocabulary: bool = False) -> Dict:
        """
        Search for relevant documents and generate a story
        
        Args:
            keywords: Keywords for story generation
            story_length: Length of story (short, medium, long)
            n_results: Number of documents to retrieve for context
            use_only_rag_vocabulary: Whether to use only words from RAG database
            
        Returns:
            Dictionary containing the generated story and metadata
        """
        try:
            # Search for relevant documents
            search_results = self.vector_db.search(keywords, n_results=n_results)
            
            if not search_results:
                print("No relevant documents found in the database.")
                # Generate story without context
                available_vocabulary = None
                if use_only_rag_vocabulary:
                    available_vocabulary = self.vector_db.get_filtered_vocabulary(keywords, [])
                
                story_result = self.story_generator.generate_story(
                    keywords, [], story_length, use_only_rag_vocabulary, available_vocabulary
                )
                
                return {
                    'story': story_result['story'],
                    'keywords': keywords,
                    'context_used': [],
                    'search_results_count': 0,
                    'generation_method': story_result.get('method', 'unknown'),
                    'keywords_used': story_result.get('keywords_used', []),
                    'keyword_usage_rate': story_result.get('keyword_usage_rate', 0),
                    'vocabulary_restricted': story_result.get('vocabulary_restricted', False),
                    'vocabulary_count': story_result.get('vocabulary_count', 0)
                }
            
            # Extract documents for context
            context_documents = [result['document'] for result in search_results]
            
            # Get filtered vocabulary if requested
            available_vocabulary = None
            if use_only_rag_vocabulary:
                available_vocabulary = self.vector_db.get_filtered_vocabulary(keywords, context_documents)
            
            # Generate story
            story_result = self.story_generator.generate_story(
                keywords, context_documents, story_length, use_only_rag_vocabulary, available_vocabulary
            )
            
            return {
                'story': story_result['story'],
                'keywords': keywords,
                'context_used': context_documents,
                'search_results_count': len(search_results),
                'search_results': search_results,
                'generation_method': story_result.get('method', 'unknown'),
                'keywords_used': story_result.get('keywords_used', []),
                'keyword_usage_rate': story_result.get('keyword_usage_rate', 0),
                'vocabulary_restricted': story_result.get('vocabulary_restricted', False),
                'vocabulary_count': story_result.get('vocabulary_count', 0),
                'context_documents_count': story_result.get('context_documents_count', 0)
            }
            
        except Exception as e:
            print(f"Error generating story: {e}")
            return {
                'story': f"Sorry, I couldn't generate a story. Error: {e}",
                'keywords': keywords,
                'context_used': [],
                'search_results_count': 0,
                'generation_method': 'error'
            }
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the vector database
        
        Returns:
            Database statistics
        """
        return self.vector_db.get_collection_info()
    
    def clear_database(self):
        """Clear all data from the vector database"""
        self.vector_db.clear_collection()
        print("Database cleared successfully!")
    
    def interactive_story_generation(self):
        """
        Interactive mode for story generation
        """
        print("\n=== RAG Story Generator ===")
        print("Type 'quit' to exit, 'stats' to see database info, 'clear' to clear database")
        
        while True:
            try:
                user_input = input("\nEnter keywords for story generation: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    stats = self.get_database_stats()
                    print(f"Database contains {stats['count']} documents")
                    continue
                elif user_input.lower() == 'clear':
                    confirm = input("Are you sure you want to clear the database? (y/n): ")
                    if confirm.lower() == 'y':
                        self.clear_database()
                    continue
                elif not user_input:
                    print("Please enter some keywords.")
                    continue
                
                # Ask for story length
                length_input = input("Story length (short/medium/long) [medium]: ").strip().lower()
                if length_input not in ['short', 'medium', 'long']:
                    length_input = 'medium'
                
                print("\nGenerating story...")
                result = self.search_and_generate_story(user_input, length_input)
                
                print(f"\n{'='*50}")
                print(f"STORY (based on keywords: {result['keywords']})")
                print(f"{'='*50}")
                print(result['story'])
                print(f"\n{'='*50}")
                print(f"Context: Used {result['search_results_count']} relevant documents")
                print(f"{'='*50}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

# Convenience function
def create_rag_system(use_openai: bool = True) -> RAGSystem:
    """
    Create and return a RAG system instance
    
    Args:
        use_openai: Whether to use OpenAI API for story generation
        
    Returns:
        RAGSystem instance
    """
    return RAGSystem(use_openai=use_openai) 