import os
from dotenv import load_dotenv
from user_manager import UserManager

# Загружаем переменные окружения
load_dotenv()

def create_admin_account():
    # Инициализируем UserManager
    user_manager = UserManager()
    
    # Данные администратора
    admin_data = {
        "username": "admin",
        "password": "admin123",  # В реальном приложении используйте более сложный пароль
        "email": "admin@example.com"
    }
    
    try:
        # Создаем аккаунт администратора
        user = user_manager.create_user(
            username=admin_data["username"],
            password=admin_data["password"],
            email=admin_data["email"]
        )
        
        # Обновляем настройки, добавляя роль администратора
        user["settings"]["role"] = "admin"
        user_manager.update_user(user["user_id"], user)
        
        print(f"Admin account created successfully!")
        print(f"Username: {admin_data['username']}")
        print(f"Password: {admin_data['password']}")
        print(f"Email: {admin_data['email']}")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    create_admin_account()
