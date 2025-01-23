import os
from typing import List, Dict, Any
import anthropic
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import logging
import uuid

class RAGAssistant:
    def __init__(self):
        # Инициализация Pinecone
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # Подключение к индексу
        self.index_name = "claude-rag"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        logging.info(f"Connected to Pinecone index: {self.index_name}")
        
        # Загрузка модели для эмбеддингов
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Инициализация Claude
        self.claude = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )

    def add_documents(self, documents: List[Dict[str, Any]], namespace: str = "documents") -> None:
        """Добавление документов в базу знаний"""
        for doc in documents:
            # Получаем эмбеддинг для документа
            embedding = self.model.encode(doc["content"]).tolist()
            
            # Добавляем документ в Pinecone
            self.index.upsert(
                vectors=[{
                    "id": doc.get("id", str(uuid.uuid4())),
                    "values": embedding,
                    "metadata": {
                        "content": doc["content"],
                        "source": doc.get("source", "unknown"),
                        "type": doc.get("type", "text")
                    }
                }],
                namespace=namespace
            )

    def search_documents(self, query: str, top_k: int = 3, namespace: str = "documents") -> List[Dict]:
        """Поиск релевантных документов"""
        # Получаем эмбеддинг для запроса
        query_embedding = self.model.encode(query).tolist()
        
        # Ищем похожие документы
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        # Форматируем результаты
        documents = []
        for match in results.matches:
            if match.metadata:
                documents.append({
                    "content": match.metadata["content"],
                    "source": match.metadata.get("source", "unknown"),
                    "score": match.score
                })
        
        return documents

    def get_answer(self, query: str, context: List[Dict] = None) -> Dict:
        """Получение ответа от Claude с учетом контекста"""
        # Формируем промпт
        system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context."
        
        if context:
            context_text = "\n\n".join([f"Source: {doc['source']}\nContent: {doc['content']}" for doc in context])
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
        else:
            user_message = query

        # Получаем ответ от Claude
        response = self.claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        return {
            "answer": response.content[0].text,
            "sources": [doc["source"] for doc in (context or [])]
        }

    def query(self, query: str, use_rag: bool = True) -> Dict:
        """Обработка запроса с использованием RAG"""
        if use_rag:
            # Поиск релевантных документов
            context = self.search_documents(query)
            # Получение ответа с учетом контекста
            result = self.get_answer(query, context)
        else:
            # Получение ответа без контекста
            result = self.get_answer(query)
        
        return result

    def clear_documents(self, namespace: str = "documents") -> None:
        """Очистка всех документов"""
        self.index.delete(delete_all=True, namespace=namespace)
