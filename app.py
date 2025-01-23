import os
import logging
from flask import Flask, request, jsonify, session, render_template, send_from_directory
from werkzeug.utils import secure_filename
from finetune import ClaudeFineTuner
from rag import RAGAssistant
from user_manager import UserManager
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any
import anthropic
import voyageai
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Track uploaded files
uploaded_files = {
    'json': None,
    'xlsx': None
}

# Создаем экземпляры классов
rag_assistant = RAGAssistant()
user_manager = UserManager()
fine_tuner = ClaudeFineTuner()

# Гибридный ассистент
class HybridAssistant:
    def __init__(self):
        self.rag = rag_assistant
        self.fine_tuner = fine_tuner
        
    def query(self, user_id: str, message: str) -> Dict:
        # Получаем настройки пользователя
        user = user_manager.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
            
        settings = user.get('settings', {})
        use_rag = settings.get('rag_enabled', True)
        use_fine_tuning = settings.get('fine_tuning_enabled', True)
        
        # Получаем ответ в зависимости от настроек
        if use_rag and use_fine_tuning:
            # Используем оба метода
            rag_response = self.rag.query(message)
            ft_response = self.fine_tuner.query(message)
            
            # Комбинируем ответы (можно улучшить логику)
            response = {
                "answer": f"RAG: {rag_response['answer']}\n\nFine-tuned: {ft_response['answer']}",
                "sources": rag_response.get('sources', []),
                "confidence": max(
                    rag_response.get('confidence', 0),
                    ft_response.get('confidence', 0)
                )
            }
        elif use_rag:
            # Только RAG
            response = self.rag.query(message)
        elif use_fine_tuning:
            # Только Fine-tuning
            response = self.fine_tuner.query(message)
        else:
            # Базовый режим без RAG и Fine-tuning
            response = self.rag.get_answer(message)
            
        return response

    def chat(self, message: str) -> Dict:
        # Получаем ответ от гибридного ассистента
        response = {
            "answer": message,
            "sources": [],
            "confidence": 0
        }
        
        return response

hybrid_assistant = HybridAssistant()

@app.before_request
def check_auth():
    """Проверка аутентификации перед каждым запросом"""
    # Публичные роуты, не требующие аутентификации
    public_routes = [
        'index',
        'static_files',
        'login',
        'register',
        'delete_all_users'
    ]
    
    if request.endpoint in public_routes:
        return
        
    # Проверяем только для защищенных роутов
    if 'user_id' not in session:
        return jsonify({"error": "Please log in to access this feature"}), 401

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        logging.info(f"Registration request data: {data}")
        
        if not data or not all(key in data for key in ['username', 'password', 'email']):
            logging.error("Missing required fields in registration data")
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
            
        try:
            user = user_manager.create_user(
                username=data['username'],
                password=data['password'],
                email=data['email']
            )
            logging.info(f"User created successfully: {user['username']}")
            
            # Устанавливаем сессию после регистрации
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            
            return jsonify({
                'success': True,
                'user': {
                    'username': user['username'],
                    'email': user['email']
                }
            })
            
        except ValueError as e:
            logging.error(f"ValueError during registration: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 400
            
    except Exception as e:
        logging.error(f"Unexpected error during registration: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Аутентификация пользователя"""
    try:
        data = request.get_json()
        logging.info(f"Login attempt for user: {data.get('username')}")
        
        if not data or not all(key in data for key in ['username', 'password']):
            logging.error("Missing required fields in login data")
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        user = user_manager.authenticate_user(
            username=data['username'],
            password=data['password']
        )
        
        if user:
            logging.info(f"User authenticated successfully: {user['username']}")
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            return jsonify({
                'success': True,
                'user': {
                    'username': user['username'],
                    'email': user['email']
                }
            })
        else:
            logging.warning(f"Authentication failed for user: {data.get('username')}")
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        logging.error(f"Error during login: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Выход пользователя"""
    session.pop('user_id', None)
    return jsonify({"message": "Logged out successfully"})

@app.route('/check_auth')
def check_auth_status():
    """Проверка статуса аутентификации"""
    if 'user_id' in session:
        user = user_manager.get_user_by_id(session['user_id'])
        if user:
            return jsonify({
                'logged_in': True,
                'username': user.get('username', 'User')
            })
    return jsonify({'logged_in': False})

@app.route('/user/settings', methods=['GET', 'PUT'])
def user_settings():
    """Получение и обновление настроек пользователя"""
    user_id = session['user_id']
    
    if request.method == 'GET':
        user = user_manager.get_user_by_id(user_id)
        if user:
            return jsonify(user['settings'])
        return jsonify({"error": "User not found"}), 404
    
    elif request.method == 'PUT':
        data = request.get_json()
        try:
            user = user_manager.update_user_settings(user_id, data)
            return jsonify(user['settings'])
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route('/user/history', methods=['GET', 'DELETE'])
def chat_history():
    """Управление историей диалогов"""
    user_id = session['user_id']
    
    if request.method == 'GET':
        history = user_manager.get_user_history(user_id)
        return jsonify(history)
    
    elif request.method == 'DELETE':
        success = user_manager.clear_chat_history(user_id)
        if success:
            return jsonify({"message": "History cleared successfully"})
        return jsonify({"error": "Failed to clear history"}), 500

@app.route('/')
def index():
    """Главная страница"""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def static_files(path):
    """Обслуживание статических файлов"""
    return send_from_directory('static', path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload/<file_type>', methods=['POST'])
def upload_file(file_type):
    try:
        logger.info(f"Receiving {file_type} file upload")
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'success': False, 'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'success': False, 'error': 'No selected file'})
        
        if file_type not in ['json', 'xlsx']:
            logger.error(f"Invalid file type: {file_type}")
            return jsonify({'success': False, 'error': 'Invalid file type'})
        
        # Validate file type
        if file_type == 'json' and not file.filename.endswith('.json'):
            logger.error("File must be a JSON file")
            return jsonify({'success': False, 'error': 'File must be a JSON file'})
        if file_type == 'xlsx' and not file.filename.endswith('.xlsx'):
            logger.error("File must be an XLSX file")
            return jsonify({'success': False, 'error': 'File must be an XLSX file'})
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files[file_type] = file_path
            logger.info(f"Successfully saved {file_type} file: {filename}")
            return jsonify({'success': True, 'filename': filename})
        
        logger.error("File upload failed")
        return jsonify({'success': False, 'error': 'File upload failed'})
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    try:
        logger.info("Starting fine-tuning process")
        if not all(uploaded_files.values()):
            logger.error("Missing files for fine-tuning")
            return jsonify({'success': False, 'error': 'Please upload both JSON and XLSX files first'})
        
        # Check if we're resuming from a checkpoint
        resume = request.args.get('resume', 'true').lower() == 'true'
        
        logger.info("Loading JSON data")
        json_data = fine_tuner.load_json_data(uploaded_files['json'])
        logger.info("Loading XLSX data")
        xlsx_data = fine_tuner.load_xlsx_data(uploaded_files['xlsx'])
        
        logger.info("Preparing training data")
        training_data = fine_tuner.prepare_training_data(json_data, xlsx_data)
        logger.info(f"Starting fine-tuning (resume={resume})")
        result = fine_tuner.fine_tune(training_data, resume=resume)
        
        logger.info(f"Fine-tuning completed. Processed {result['successful_batches']}/{result['total_batches']} batches")
        return jsonify({
            'success': True, 
            'message': f"Fine-tuning complete. Processed {result['successful_batches']}/{result['total_batches']} batches successfully.",
            'response': {
                'total_batches': result['total_batches'],
                'successful_batches': result['successful_batches']
            }
        })
    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_documents', methods=['POST'])
def upload_documents():
    try:
        if 'documents' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
            
        files = request.files.getlist('documents')
        documents = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                # Определяем тип файла по расширению
                ext = file.filename.rsplit('.', 1)[1].lower()
                
                try:
                    if ext in {'txt', 'csv', 'json'}:
                        # Текстовые файлы читаем как UTF-8
                        content = file.read().decode('utf-8')
                    else:
                        # Бинарные файлы (PDF, DOC, DOCX) читаем как base64
                        import base64
                        content = base64.b64encode(file.read()).decode('utf-8')
                        
                    documents.append({
                        'title': file.filename,
                        'content': content,
                        'source': 'upload',
                        'type': ext,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    continue
                
        if not documents:
            return jsonify({'success': False, 'error': 'No valid documents found'})
            
        # Initialize RAG system
        rag = RAGAssistant()
        
        # Add documents to knowledge base
        results = rag.batch_add_documents(documents)
        
        return jsonify({
            'success': True,
            'added': results['success'],
            'failed': results['failed']
        })
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_kb', methods=['POST'])
def clear_knowledge_base():
    try:
        rag = RAGAssistant()
        success = rag.clear_knowledge_base()
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/rag', methods=['POST'])
def rag_query():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
            
        rag = RAGAssistant()
        result = rag.query(query)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_index', methods=['POST'])
def delete_index():
    try:
        rag = RAGAssistant()
        success = rag.delete_index()
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_files():
    """Загрузка файлов для fine-tuning"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
            
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({'success': False, 'error': 'No files selected'}), 400

        # Создаем директорию для загрузок, если её нет
        upload_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        uploaded_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)

        if not uploaded_files:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400

        # Запускаем fine-tuning
        fine_tuner.train(uploaded_files)

        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'files': uploaded_files
        })

    except Exception as e:
        logging.error(f"Error uploading files: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def process_query():
    """Обработка RAG запроса"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'No query provided'}), 400

        # Получаем ответ от RAG
        response = rag_assistant.process_query(data['query'])
        
        return jsonify({
            'success': True,
            'answer': response
        })

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Обработка сообщения в чате"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400

        # Получаем ответ от гибридного ассистента
        response = hybrid_assistant.chat(data['message'])
        
        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        logging.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/make_admin/<username>', methods=['POST'])
def make_admin(username):
    try:
        # Получаем пользователя по имени
        user = user_manager.get_user_by_username(username)
        if not user:
            return jsonify({'success': False, 'error': 'User not found'})
            
        # Обновляем настройки пользователя
        settings = user.get('settings', {})
        settings['role'] = 'admin'
        user_manager.update_user_settings(user['user_id'], settings)
        
        return jsonify({
            'success': True,
            'message': f'User {username} is now an administrator'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_all_users', methods=['POST'])
def delete_all_users():
    """Удаление всех пользователей"""
    try:
        # Отключаем проверку аутентификации для этого роута
        success = user_manager.delete_all_users()
        if success:
            # Очищаем сессию
            session.clear()
            return jsonify({'success': True, 'message': 'All users deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete users'}), 500
    except Exception as e:
        logging.error(f"Error deleting all users: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload_rag', methods=['POST'])
def upload_rag():
    """Загрузка документов для RAG"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
            
        files = request.files.getlist('files[]')
        if not files:
            return jsonify({'success': False, 'error': 'No files selected'}), 400

        # Создаем директорию для RAG документов, если её нет
        rag_dir = os.path.join(app.root_path, 'rag_documents')
        os.makedirs(rag_dir, exist_ok=True)

        uploaded_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(rag_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)

        if not uploaded_files:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400

        # Добавляем документы в RAG
        for filename in uploaded_files:
            filepath = os.path.join(rag_dir, filename)
            rag_assistant.add_document(filepath)

        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} documents',
            'files': uploaded_files
        })

    except Exception as e:
        logging.error(f"Error uploading RAG documents: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
