import os
import json
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import uuid
import numpy as np
import logging

class UserManager:
    def __init__(self):
        # Инициализация Pinecone
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # Используем индекс для пользователей
        self.index_name = "hybrid-assistant"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # Используем то же измерение, что и для RAG
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
                )
            )
        self.index = self.pc.Index(self.index_name)
        logging.info(f"Connected to Pinecone index: {self.index_name}")

    def _hash_password(self, password: str) -> str:
        """Хеширование пароля"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _create_random_vector(self) -> List[float]:
        """Создает случайный вектор"""
        return np.random.rand(1536).astype(np.float32).tolist()

    def create_user(self, username: str, password: str, email: str) -> Dict:
        """Создание нового пользователя"""
        try:
            # Проверяем, существует ли пользователь с таким email
            existing = self.get_user_by_username(email)
            if existing:
                raise ValueError("Email already registered")

            user_id = str(uuid.uuid4())
            
            # Создаем настройки по умолчанию
            settings = {
                "rag_enabled": True,
                "fine_tuning_enabled": True,
                "theme": "light",
                "role": "user"
            }

            user_data = {
                "user_id": user_id,
                "username": username,
                "password_hash": self._hash_password(password),
                "email": email,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": "",
                "settings": json.dumps(settings)  # Сохраняем настройки как JSON строку
            }

            # Сохраняем в Pinecone с namespace для пользователей
            logging.info(f"Creating user with email: {email}")
            vector_id = f"user_{user_id}"
            vector_data = {
                "id": vector_id,
                "values": self._create_random_vector(),
                "metadata": user_data
            }
            logging.info(f"Upserting vector with metadata: {user_data}")
            
            self.index.upsert(
                vectors=[vector_data],
                namespace="users"
            )
            
            # Проверяем, что пользователь создался
            describe = self.index.describe_index_stats()
            logging.info(f"Index stats after user creation: {describe}")
            
            # Пробуем найти созданного пользователя
            check_user = self.get_user_by_username(email)
            if check_user:
                logging.info(f"Successfully verified user creation: {check_user}")
            else:
                logging.warning(f"User creation verification failed - user not found: {email}")
                # Пробуем получить вектор напрямую
                fetch_result = self.index.fetch(
                    ids=[vector_id],
                    namespace="users"
                )
                logging.info(f"Direct vector fetch result: {fetch_result}")

            logging.info(f"User created successfully: {email}")

            # Возвращаем данные с распакованными настройками
            user_data["settings"] = settings
            return user_data
            
        except Exception as e:
            logging.error(f"Error creating user: {str(e)}", exc_info=True)
            raise

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Получение пользователя по email"""
        try:
            logging.info(f"Searching for user by email: {username}")
            
            # Получаем список всех пользователей
            describe = self.index.describe_index_stats()
            logging.info(f"Index stats: {describe}")
            
            # Выполняем поиск
            results = self.index.query(
                vector=self._create_random_vector(),
                filter={"email": {"$eq": username}},
                namespace="users",
                top_k=1,
                include_metadata=True  # Важно: включаем метаданные в ответ
            )
            
            logging.info(f"Query results: {results.matches}")
            
            if results.matches:
                match = results.matches[0]
                if hasattr(match, 'metadata') and match.metadata:
                    user_data = match.metadata
                    logging.info(f"Found user data: {user_data}")
                    # Распаковываем настройки из JSON
                    if isinstance(user_data.get('settings'), str):
                        user_data["settings"] = json.loads(user_data["settings"])
                    logging.info(f"Found user: {user_data}")
                    return user_data
                else:
                    logging.warning(f"Match found but no metadata: {match}")
            logging.warning(f"No user found with email: {username}")
            return None
            
        except Exception as e:
            logging.error(f"Error getting user by email: {str(e)}", exc_info=True)
            raise

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Получение пользователя по ID"""
        try:
            results = self.index.fetch(
                ids=[f"user_{user_id}"],
                namespace="users"
            )
            
            if results and f"user_{user_id}" in results["vectors"]:
                user_data = results["vectors"][f"user_{user_id}"].metadata
                # Распаковываем настройки из JSON
                if isinstance(user_data.get('settings'), str):
                    user_data["settings"] = json.loads(user_data["settings"])
                return user_data
            return None
            
        except Exception as e:
            logging.error(f"Error getting user by ID: {str(e)}", exc_info=True)
            raise

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Аутентификация пользователя"""
        try:
            logging.info(f"Authenticating user: {username}")
            user = self.get_user_by_username(username)
            
            if not user:
                logging.warning(f"User not found: {username}")
                return None
                
            if user["password_hash"] == self._hash_password(password):
                logging.info(f"Password correct for user: {username}")
                # Обновляем время последнего входа
                user["last_login"] = datetime.utcnow().isoformat()
                self.update_user(user["user_id"], user)
                return user
            else:
                logging.warning(f"Invalid password for user: {username}")
                return None
            
        except Exception as e:
            logging.error(f"Error authenticating user: {str(e)}", exc_info=True)
            raise

    def update_user(self, user_id: str, user_data: Dict) -> Dict:
        """Обновление данных пользователя"""
        try:
            # Если настройки являются словарем, преобразуем их в JSON
            if isinstance(user_data["settings"], dict):
                user_data["settings"] = json.dumps(user_data["settings"])
                
            self.index.upsert(
                vectors=[{
                    "id": f"user_{user_id}",
                    "values": self._create_random_vector(),
                    "metadata": user_data
                }],
                namespace="users"
            )
            
            # Возвращаем данные с распакованными настройками
            user_data["settings"] = json.loads(user_data["settings"])
            return user_data
            
        except Exception as e:
            logging.error(f"Error updating user: {str(e)}", exc_info=True)
            raise

    def update_user_settings(self, user_id: str, settings: Dict) -> Dict:
        """Обновление настроек пользователя"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                raise ValueError("User not found")
            
            # Обновляем настройки
            current_settings = user["settings"]
            current_settings.update(settings)
            user["settings"] = current_settings
            
            return self.update_user(user_id, user)
            
        except Exception as e:
            logging.error(f"Error updating user settings: {str(e)}", exc_info=True)
            raise

    def delete_user(self, user_id: str) -> bool:
        """Удаление пользователя"""
        try:
            self.index.delete(
                ids=[f"user_{user_id}"],
                namespace="users"
            )
            return True
        except Exception as e:
            logging.error(f"Error deleting user: {str(e)}", exc_info=True)
            return False

    def delete_all_users(self) -> bool:
        """Удаление всех пользователей"""
        try:
            # Получаем всех пользователей
            results = self.index.query(
                vector=self._create_random_vector(),
                namespace="users",
                top_k=1000
            )
            
            # Собираем ID всех пользователей
            user_ids = [match.id for match in results.matches]
            
            if user_ids:
                # Удаляем всех пользователей
                self.index.delete(
                    ids=user_ids,
                    namespace="users"
                )
                logging.info(f"Deleted {len(user_ids)} users")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting all users: {str(e)}", exc_info=True)
            return False

    def get_user_history(self, user_id: str) -> List[Dict]:
        """Получение истории диалогов пользователя"""
        try:
            results = self.index.query(
                vector=self._create_random_vector(),
                filter={"user_id": {"$eq": user_id}},
                namespace="chat_history",
                top_k=100
            )
            
            history = []
            for match in results.matches:
                if match.metadata:
                    history.append(match.metadata)
            
            return sorted(history, key=lambda x: x.get('timestamp', ''))
            
        except Exception as e:
            logging.error(f"Error getting user history: {str(e)}", exc_info=True)
            raise

    def add_chat_history(self, user_id: str, message: Dict) -> Dict:
        """Добавление записи в историю диалогов"""
        try:
            message_id = str(uuid.uuid4())
            message_data = {
                "message_id": message_id,
                "user_id": user_id,
                "content": message["content"],
                "role": message["role"],
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": json.dumps(message.get("metadata", {}))  # Сохраняем метаданные как JSON
            }

            self.index.upsert(
                vectors=[{
                    "id": f"msg_{message_id}",
                    "values": self._create_random_vector(),
                    "metadata": message_data
                }],
                namespace="chat_history"
            )

            return message_data
            
        except Exception as e:
            logging.error(f"Error adding chat history: {str(e)}", exc_info=True)
            raise

    def clear_chat_history(self, user_id: str) -> bool:
        """Очистка истории диалогов пользователя"""
        try:
            results = self.index.query(
                vector=self._create_random_vector(),
                filter={"user_id": {"$eq": user_id}},
                namespace="chat_history",
                top_k=1000
            )
            
            message_ids = [f"msg_{match.metadata['message_id']}" for match in results.matches]
            if message_ids:
                self.index.delete(
                    ids=message_ids,
                    namespace="chat_history"
                )
            return True
            
        except Exception as e:
            logging.error(f"Error clearing chat history: {str(e)}", exc_info=True)
            return False
