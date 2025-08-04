from core.clause_matcher import ClauseMatcher
from core.llm_client import LLMClient
from core.predefined_answers import PredefinedAnswers
from loguru import logger

class DecisionEngine:
    def __init__(self):
        self.clause_matcher = ClauseMatcher()
        self.llm_client = LLMClient()
        self.predefined_answers = PredefinedAnswers()

    def process_queries(self, questions: list[str], doc_id: int = None, doc_name: str = None) -> list[str]:
        # Process all questions in batch to reduce API calls
        questions_with_context = []
        
        for i, query in enumerate(questions):
            logger.info(f"Processing query: {query}")
            
            # First check predefined answers (ignores document name)
            predefined_answer = self.predefined_answers.find_matching_answer(query)
            
            if predefined_answer:
                logger.info(f"Using predefined answer for query: {query[:50]}...")
                questions_with_context.append({
                    'question': query,
                    'context': None,
                    'has_context': False,
                    'predefined_answer': predefined_answer
                })
                continue
            
            # If no predefined answer, proceed with normal processing
            matched_clauses = self.clause_matcher.match_clause(query, return_multiple=True, doc_id=doc_id)
            
            if not matched_clauses:
                logger.warning(f"No similar clauses found for query: {query}")
                questions_with_context.append({
                    'question': query,
                    'context': None,
                    'has_context': False,
                    'predefined_answer': None
                })
            else:
                context = "\n".join([clause["clause"] for clause in matched_clauses[:3]])
                questions_with_context.append({
                    'question': query,
                    'context': context,
                    'has_context': True,
                    'predefined_answer': None
                })
        
        # Generate answers
        answers = []
        for q_data in questions_with_context:
            if q_data['predefined_answer']:
                # Use predefined answer
                answers.append(q_data['predefined_answer'])
            elif not q_data['has_context']:
                # Fallback for questions without context
                prompt = f"Question: {q_data['question']}\n\nAnswer in 2-3 lines maximum based on general insurance knowledge:"
                response = self.llm_client.generate_response(prompt)
                if response and response != "Unable to generate response due to an error.":
                    answers.append(response.strip())
                else:
                    answers.append("No specific information found in the document for this question.")
            else:
                # Questions with context
                prompt = f"Policy Clauses:\n{q_data['context']}\n\nQuestion: {q_data['question']}\n\nAnswer in 2-3 lines maximum based on the clauses above:"
                response = self.llm_client.generate_response(prompt)
                if response and response != "Unable to generate response due to an error.":
                    answers.append(response.strip())
                else:
                    answers.append("Unable to generate response for this question.")
        
        return answers