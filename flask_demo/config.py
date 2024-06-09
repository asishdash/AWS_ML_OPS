

class Config:
    DEBUG = True
    TESTING = False
    DATABASE_URI = 'sqlite:///my_database.db'
    SECRET_KEY = 'supersecretkey'
    API_KEY = 'your_api_key_here'
class ProductionConfig(Config):
    DEBUG = False
    DATABASE_URI = 'sqlite:///prod_database.db'
class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URI = "sqlite:///dev_database.db"
class TestingConfig(Config):
    TESTING = True
    DATABASE_URI = 'sqlite:///test_database.db'