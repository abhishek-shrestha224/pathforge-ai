class Config(object):
  TESTING = False
  DEBUG = False
  FLASK_ENV = "development"
  FLASK_APP = "run"


class ProductionConfig(Config):
  FLASK_ENV = "production"


class DevelopmentConfig(Config):
  DEBUG = True


class TestingConfig(Config):
  TESTING = True
