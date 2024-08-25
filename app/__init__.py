from flask import Flask
from config_module import ProductionConfig

app = Flask(__name__)
app.config.from_object(ProductionConfig())

from app import views
