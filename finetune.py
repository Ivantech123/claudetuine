import os
import json
import time
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import logging
import pickle
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ModelEvaluator:
    """
    Класс для оценки качества обучения модели
    """
    def __init__(self):
        self.metrics_history = {
            'perplexity': [],
            'uncertainty': [],
            'completeness': [],
            'context_relevance': []
        }
        
    def calculate_perplexity(self, probabilities):
        """
        Вычисляет перплексию на основе вероятностей
        Меньшее значение = лучше
        """
        return np.exp(-np.mean(np.log(probabilities)))
        
    def calculate_uncertainty(self, log_probs):
        """
        Оценивает неопределенность модели
        Меньшее значение = лучше
        """
        return -np.mean(log_probs)
        
    def evaluate_completeness(self, response, prompt):
        """
        Оценивает полноту ответа
        """
        # Базовая оценка на основе длины ответа и наличия ключевых элементов
        if not response or len(response) < 10:
            return 0.0
            
        # Проверяем наличие ключевых элементов из промпта
        prompt_keywords = set(word.lower() for word in prompt.split())
        response_keywords = set(word.lower() for word in response.split())
        
        overlap = len(prompt_keywords.intersection(response_keywords))
        completeness = overlap / len(prompt_keywords) if prompt_keywords else 0.0
        
        return min(1.0, completeness)
        
    def evaluate_context_relevance(self, response, context):
        """
        Оценивает релевантность ответа контексту
        """
        if not response or not context:
            return 0.0
            
        # Простая оценка на основе пересечения слов
        context_words = set(word.lower() for word in context.split())
        response_words = set(word.lower() for word in response.split())
        
        overlap = len(context_words.intersection(response_words))
        relevance = overlap / len(context_words) if context_words else 0.0
        
        return min(1.0, relevance)
        
    def update_metrics(self, batch_metrics):
        """
        Обновляет историю метрик
        """
        for metric, value in batch_metrics.items():
            if metric in self.metrics_history:
                self.metrics_history[metric].append(value)
                
    def get_average_metrics(self):
        """
        Возвращает средние значения метрик
        """
        return {
            metric: np.mean(values) if values else 0.0
            for metric, values in self.metrics_history.items()
        }
        
    def should_stop_training(self, patience=3, min_improvement=0.01):
        """
        Проверяет, нужно ли остановить обучение
        """
        if len(self.metrics_history['perplexity']) < patience:
            return False
            
        # Проверяем улучшение перплексии
        recent_perplexity = self.metrics_history['perplexity'][-patience:]
        improvements = [recent_perplexity[i-1] - recent_perplexity[i] 
                       for i in range(1, len(recent_perplexity))]
                       
        # Если улучшения меньше порога, рекомендуем остановку
        return all(imp < min_improvement for imp in improvements)

class ClaudeFineTuner:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.MAX_MESSAGES = 900
        # Динамический размер батча будет установлен позже
        self.BATCH_SIZE = None
        self.learning_rate = 1.0  # Значение по умолчанию
        self.checkpoint_file = 'fine_tuning_checkpoint.pkl'
        self.metrics_history = []  # История метрик для отслеживания прогресса
        self.evaluator = ModelEvaluator()  # Добавляем оценщик
        
    def _determine_batch_size(self, dataset_size):
        """
        Определяет оптимальный размер батча на основе размера датасета
        """
        if dataset_size >= 1000:
            return 64  # Оптимально для больших датасетов
        elif 500 <= dataset_size < 1000:
            return 32  # Средний размер датасета
        else:
            return 16  # Малый размер датасета

    def _determine_learning_rate(self, dataset_size):
        """
        Определяет оптимальный learning rate на основе размера датасета
        """
        if dataset_size >= 1000:
            return 1.5  # Повышенный learning rate для больших датасетов
        elif 500 <= dataset_size < 1000:
            return 1.0  # Стандартный learning rate
        else:
            return 0.5  # Пониженный learning rate для малых датасетов

    def validate_data_quality(self, data_item):
        """
        Проверяет качество отдельного элемента данных
        Возвращает (bool, str) - (прошел ли проверку, причина если нет)
        """
        if not isinstance(data_item, dict):
            return False, "Элемент данных должен быть словарем"
            
        # Проверяем наличие обязательных полей
        required_fields = ['input', 'output']
        for field in required_fields:
            if field not in data_item:
                return False, f"Отсутствует обязательное поле: {field}"
                
        # Проверяем, что поля не пустые
        if not data_item['input'].strip() or not data_item['output'].strip():
            return False, "Поля input/output не могут быть пустыми"
            
        # Проверяем длину текста (можно настроить лимиты)
        if len(data_item['input']) < 10:
            return False, "Входной текст слишком короткий"
            
        if len(data_item['output']) < 5:
            return False, "Выходной текст слишком короткий"
            
        return True, "OK"

    def load_json_data(self, json_path):
        """Load and process JSON training data"""
        try:
            logger.info(f"Loading JSON file from: {json_path}")
            # Print file contents for debugging
            with open(json_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                logger.debug(f"File contents (first 500 chars): {file_content[:500]}")
                raw_data = json.loads(file_content)
            
            logger.info(f"Raw JSON data length: {len(raw_data)}")
            logger.debug(f"First item in raw data: {raw_data[0] if raw_data else 'No data'}")
            
            # Check data structure
            if isinstance(raw_data, dict):
                logger.debug(f"JSON root is a dictionary with keys: {list(raw_data.keys())}")
                # Handle case where data might be nested
                if 'data' in raw_data:
                    raw_data = raw_data['data']
                elif 'messages' in raw_data:
                    raw_data = raw_data['messages']
                elif 'conversations' in raw_data:
                    raw_data = raw_data['conversations']
                elif 'train' in raw_data:
                    raw_data = raw_data['train']
            
            if not isinstance(raw_data, list):
                logger.error(f"Invalid JSON structure. Expected list, got {type(raw_data)}")
                raw_data = []
            
            valid_data = []
            for item in raw_data:
                logger.debug(f"Processing item: {str(item)[:200]}")  # Log first 200 chars of each item
                # Try different possible key names
                input_text = None
                output_text = None
                
                if isinstance(item, dict):
                    logger.debug(f"Item keys: {list(item.keys())}")
                
                # Common key patterns for input
                input_keys = ['input', 'prompt', 'question', 'human', 'user_message', 'user', 'text']
                output_keys = ['output', 'response', 'answer', 'assistant', 'bot_message', 'completion', 'label']
                
                # Try to find input text
                for key in input_keys:
                    if key in item:
                        input_text = str(item[key]).strip()
                        logger.debug(f"Found input with key '{key}': {input_text[:100]}")
                        break
                    elif isinstance(item, dict) and any(k.lower() == key for k in item.keys()):
                        input_text = str(item[next(k for k in item.keys() if k.lower() == key)]).strip()
                        logger.debug(f"Found input with case-insensitive key '{key}': {input_text[:100]}")
                        break
                
                # Try to find output text
                for key in output_keys:
                    if key in item:
                        output_text = str(item[key]).strip()
                        logger.debug(f"Found output with key '{key}': {output_text[:100]}")
                        break
                    elif isinstance(item, dict) and any(k.lower() == key for k in item.keys()):
                        output_text = str(item[next(k for k in item.keys() if k.lower() == key)]).strip()
                        logger.debug(f"Found output with case-insensitive key '{key}': {output_text[:100]}")
                        break
                
                if input_text and output_text:
                    valid_data.append({
                        'input': input_text,
                        'output': output_text
                    })
                else:
                    logger.warning(f"Skipping item - missing input or output: {str(item)[:200]}")
                    
            logger.info(f"Extracted {len(valid_data)} valid examples from JSON")
            if valid_data:
                logger.debug(f"Sample valid item: {valid_data[0]}")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}", exc_info=True)
            return []

    def load_xlsx_data(self, xlsx_path):
        """Load and process XLSX training data"""
        try:
            logger.info(f"Loading XLSX file from: {xlsx_path}")
            df = pd.read_excel(xlsx_path)
            logger.info(f"XLSX columns: {df.columns.tolist()}")
            logger.info(f"XLSX shape: {df.shape}")
            
            valid_data = []
            
            # Try different column name patterns
            input_cols = ['input', 'prompt', 'question', 'human', 'user_message', 'user']
            output_cols = ['output', 'response', 'answer', 'assistant', 'bot_message', 'completion']
            
            # Find the actual column names
            input_col = next((col for col in df.columns if col.lower() in input_cols), None)
            output_col = next((col for col in df.columns if col.lower() in output_cols), None)
            
            if not input_col or not output_col:
                logger.error(f"Could not find input/output columns. Available columns: {df.columns.tolist()}")
                return []
            
            logger.info(f"Using columns: input='{input_col}', output='{output_col}'")
            
            for _, row in df.iterrows():
                input_text = str(row[input_col]).strip()
                output_text = str(row[output_col]).strip()
                
                if input_text and output_text and input_text.lower() != 'nan' and output_text.lower() != 'nan':
                    valid_data.append({
                        'input': input_text,
                        'output': output_text
                    })
            
            logger.info(f"Extracted {len(valid_data)} valid examples from XLSX")
            if valid_data:
                logger.debug(f"Sample valid item: {valid_data[0]}")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading XLSX data: {str(e)}", exc_info=True)
            return []

    def prepare_training_data(self, json_data, xlsx_data):
        """Подготовка данных для обучения с валидацией"""
        logger.info("Preparing training data...")
        
        valid_data = []
        invalid_count = 0
        
        # Объединяем данные из обоих источников
        all_data = []
        if json_data:
            all_data.extend(json_data)
        if xlsx_data:
            all_data.extend(xlsx_data)
            
        # Проверяем каждый элемент данных
        for item in all_data:
            is_valid, reason = self.validate_data_quality(item)
            if is_valid:
                valid_data.append(item)
            else:
                invalid_count += 1
                logger.warning(f"Invalid data item: {reason}")
                
        logger.info(f"Validation complete. Valid items: {len(valid_data)}, Invalid items: {invalid_count}")
        
        # Устанавливаем оптимальный размер батча
        self.BATCH_SIZE = self._determine_batch_size(len(valid_data))
        logger.info(f"Set batch size to {self.BATCH_SIZE} for dataset size {len(valid_data)}")
        
        return valid_data

    def batch_training_data(self, training_data):
        """Split training data into batches"""
        if not training_data:
            logger.error("No valid training data to batch")
            return []
            
        batches = []
        current_batch = []
        current_message_count = 0
        
        for item in training_data:
            if not item['input'].strip() or not item['output'].strip():
                continue
                
            if current_message_count + 2 > self.MAX_MESSAGES or len(current_batch) >= self.BATCH_SIZE:
                if current_batch:
                    batches.append(current_batch)
                current_batch = []
                current_message_count = 0
            
            current_batch.append(item)
            current_message_count += 2
        
        if current_batch:
            batches.append(current_batch)
            
        logger.info(f"Created {len(batches)} batches from {len(training_data)} examples")
        if batches:
            logger.debug(f"Sample batch size: {len(batches[0])}")
            logger.debug(f"Sample item from first batch: {batches[0][0]}")
            
        return batches

    def save_checkpoint(self, batch_idx, processed_data, metrics):
        """
        Сохраняет расширенную контрольную точку с метриками
        """
        checkpoint = {
            'batch_idx': batch_idx,
            'processed_data': processed_data,
            'learning_rate': self.learning_rate,
            'batch_size': self.BATCH_SIZE,
            'metrics': metrics,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info(f"Checkpoint saved: batch {batch_idx}, metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")

    def load_checkpoint(self):
        """
        Загружает контрольную точку с дополнительной валидацией
        """
        if not os.path.exists(self.checkpoint_file):
            return None, None, None
            
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Проверяем структуру checkpoint
            required_fields = ['batch_idx', 'processed_data', 'learning_rate', 'batch_size']
            if not all(field in checkpoint for field in required_fields):
                logger.warning("Checkpoint file has invalid structure")
                return None, None, None
                
            logger.info(f"Loaded checkpoint from {checkpoint.get('timestamp', 'unknown time')}")
            return checkpoint['batch_idx'], checkpoint['processed_data'], checkpoint['metrics']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None, None, None

    def should_stop_early(self, metrics_history, patience=3):
        """
        Проверяет, нужно ли остановить обучение рано
        """
        if len(metrics_history) < patience:
            return False
            
        # Проверяем последние метрики на ухудшение
        recent_metrics = metrics_history[-patience:]
        if all(recent_metrics[i] <= recent_metrics[i-1] for i in range(1, len(recent_metrics))):
            logger.info("Early stopping triggered: no improvement in recent iterations")
            return True
        return False

    def process_batch(self, batch):
        """Обрабатывает батч с оценкой качества"""
        try:
            messages = []
            for item in batch:
                input_text = item['input'].strip()
                output_text = item['output'].strip()
                
                if input_text and output_text:
                    messages.append({
                        "role": "user",
                        "content": input_text
                    })
                    messages.append({
                        "role": "assistant",
                        "content": output_text
                    })
                    
            if not messages:
                return None
                
            # Отправляем запрос к API
            response = self.anthropic.beta.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=messages
            )
            
            # Оцениваем качество ответа
            metrics = {
                'perplexity': self.evaluator.calculate_perplexity(response.logprobs),
                'uncertainty': self.evaluator.calculate_uncertainty(response.logprobs),
                'completeness': np.mean([
                    self.evaluator.evaluate_completeness(msg['content'], messages[i-1]['content'])
                    for i, msg in enumerate(messages) if msg['role'] == 'assistant'
                ]),
                'context_relevance': np.mean([
                    self.evaluator.evaluate_context_relevance(msg['content'], messages[i-1]['content'])
                    for i, msg in enumerate(messages) if msg['role'] == 'assistant'
                ])
            }
            
            # Обновляем метрики
            self.evaluator.update_metrics(metrics)
            
            return {
                'response': response,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return None

    def fine_tune(self, training_data, resume=True):
        """Выполняет fine-tuning модели с оптимизированными параметрами"""
        
        # Определяем оптимальные параметры
        self.learning_rate = self._determine_learning_rate(len(training_data))
        if self.BATCH_SIZE is None:
            self.BATCH_SIZE = self._determine_batch_size(len(training_data))
            
        logger.info(f"Starting fine-tuning with learning_rate={self.learning_rate}, batch_size={self.BATCH_SIZE}")
        
        # Загружаем чекпоинт если нужно продолжить
        start_batch, processed_data, last_metrics = self.load_checkpoint() if resume else (0, [], None)
        if resume and start_batch is not None:
            logger.info(f"Resuming from batch {start_batch}")
            if last_metrics:
                self.metrics_history = last_metrics
        else:
            start_batch = 0
            processed_data = []
            
        batches = self.batch_training_data(training_data)
        total_batches = len(batches)
        successful_batches = 0
        
        try:
            for i, batch in enumerate(batches[start_batch:], start=start_batch):
                try:
                    # Обработка батча
                    result = self.process_batch(batch)
                    if result:
                        successful_batches += 1
                        processed_data.extend(batch)
                        
                        # Сохраняем метрики и проверяем early stopping
                        if 'metrics' in result:
                            self.metrics_history.append(result['metrics'])
                            if self.should_stop_early(self.metrics_history):
                                logger.info("Early stopping triggered")
                                break
                                
                    # Сохраняем чекпоинт каждые 5 батчей
                    if i % 5 == 0:
                        self.save_checkpoint(i, processed_data, self.metrics_history)
                        
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Fine-tuning interrupted by user")
            # Сохраняем финальный чекпоинт
            self.save_checkpoint(i, processed_data, self.metrics_history)
            
        return {
            'total_batches': total_batches,
            'successful_batches': successful_batches,
            'learning_rate': self.learning_rate,
            'batch_size': self.BATCH_SIZE,
            'metrics_history': self.metrics_history
        }

def main():
    # Initialize fine-tuner
    fine_tuner = ClaudeFineTuner()
    
    # Load training data
    json_data = fine_tuner.load_json_data('training_data.json')
    xlsx_data = fine_tuner.load_xlsx_data('training_data.xlsx')
    
    # Prepare training data
    training_data = fine_tuner.prepare_training_data(json_data, xlsx_data)
    
    # Fine-tune the model
    response = fine_tuner.fine_tune(training_data)
    print("Fine-tuning complete:", response)

if __name__ == "__main__":
    main()
